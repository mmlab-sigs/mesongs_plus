import os
import tempfile
import torch 
import numpy as np
import shutil
from .laplace_codec import laplace_encode_blocks, get_encoded_size
from .ntk_codec import ntk_encode
from .gpcc_codec import gpcc_encode_octree, is_gpcc_available

def get_dtype_for_bits(num_bits):
    if num_bits <= 8:
        return np.uint8
    elif num_bits <= 16:
        return np.uint16
    elif num_bits <= 32:
        return np.uint32
    else:
        return np.uint64
# from utils.quant_utils import split_length, cal_diff_square_wrapper, cal_diff_wrapper, cal_diff_infinity_wrapper, pure_quant_wo_minmax
# from utils.tools import ToEulerAngles_FT, decode_oct, to_int_bits
# from raht_torch import transform_batched_torch, copyAsort, haar3D_param, inv_haar3D_param
from .meson_utils import split_length, cal_diff_square_wrapper, cal_diff_wrapper, cal_diff_infinity_wrapper, pure_quant_wo_minmax
from .meson_utils import ToEulerAngles_FT, decode_oct
from .raht_torch import transform_batched_torch, copyAsort, haar3D_param, inv_haar3D_param
from pulp import LpProblem, LpVariable, LpMinimize, LpInteger, LpStatus, PULP_CBC_CMD

def to_int_bits(x, splits_list, qbits, channel_check, n_block):
    # Wrapper to perform quantization and return indices + scales/zps
    power_qbits = (2 ** torch.tensor(qbits, device=x.device) - 1).int()
    # splits_list is list of ints.
    # pure_quant_wo_minmax expects splits as tensor [C, n_block]
    splits_tensor = torch.tensor(splits_list, device=x.device, dtype=torch.int32).unsqueeze(0).repeat(x.shape[1], 1)
    
    q_indices, scales, zps, _, _ = pure_quant_wo_minmax(x, splits_tensor, power_qbits)
    
    # Collect scales/zps interleaved
    sz = torch.stack([scales, zps], dim=-1).flatten().cpu().tolist()
    
    # Split indices into q8 (<=8 bits) and q16 (>8 bits)
    # This logic depends on how channel_check is defined/used.
    # channel_check is boolean mask [C, B]? 
    # But usually we split by channel or save all together if mixed?
    # For size estimation, we just need to save the array in correct type.
    # If ANY block in a channel needs >8 bits, that channel column in q_indices will have large values.
    # But q_indices is [N, C].
    # We can check max value of q_indices or use qbits.
    
    # Simplifying: save as uint16 if any qbit > 8, else uint8.
    # But to separate them as the original tool seemed to intend:
    # Maybe it separated columns?
    # Let's just return all as one array for now, adjusting save_npz.
    
    # Use maximum bit width to determine appropriate dtype
    max_bit = int(qbits.max())
    q_dtype = get_dtype_for_bits(max_bit)
    
    # Save all indices in appropriate dtype
    q_out = q_indices.cpu().numpy().astype(q_dtype)
    
    # For backward compatibility with save_npz, return based on max bit width
    if max_bit > 8:
        return None, q_out, sz  # q8 is None, q16 is all (stored as appropriate dtype)
    else:
        return q_out, None, sz

def get_ddrf(
    npc, n_block, n_channel, 
    dis_type=2, low_bit=1, high_bit=8):
    
    n_len = npc.shape[0]
    splits = torch.tensor([split_length(n_len, n_block)] * n_channel, device="cuda")

    # 1: abs
    # 2: square
    # 3: inf
    if dis_type == 1:
        cal_diff_func = cal_diff_wrapper
    elif dis_type == 2:
        cal_diff_func = cal_diff_square_wrapper
    elif dis_type == 3:
        cal_diff_func = cal_diff_infinity_wrapper
    A = []
    with torch.no_grad():
        for j in range(low_bit, high_bit + 1):
            out = cal_diff_func(
                channel_num = n_channel,
                block_num = n_block,
                inputs = npc,
                splits = splits,
                qbits = torch.ones([n_channel, n_block], device="cuda")*j,
            )
            torch.cuda.synchronize()
            A.append(out.clone())
    
    A = torch.stack(A) # [n_bit, C, B]
    B = torch.sum(A, dim=-1).transpose(0, 1) 
    A = A.permute(1, 2, 0) # [C, B, n_bit]
    B_flat = B.flatten().cpu().tolist()
    return A, B_flat, npc

def qbit_channel_size_estimator(vars01, n_bit, low_bit, C, B, imp_splits_list, sen_splits_list, equal_bit_val=6, search_rf=True, search_scale=True, c_rf=7, c_scale=3):
    if search_rf and search_scale:
        qbsum = 0
        for ci in range(0, C):
            for bi in range(0, B):
                for j in range(n_bit):
                    if C < 8:
                        qbsum += vars01[ci * B * n_bit + bi * n_bit + j] * (j + low_bit) * imp_splits_list[bi]
                    else:
                        qbsum += vars01[ci * B * n_bit + bi * n_bit + j] * (j + low_bit) * sen_splits_list[bi]
    elif search_rf:
        qbsum = 0
        for ci in range(0, C):
            for bi in range(0, B):
                for j in range(n_bit):
                        qbsum += vars01[ci * B * n_bit + bi * n_bit + j] * (j + low_bit) * imp_splits_list[bi]
        
        qbsum += equal_bit_val * sum(sen_splits_list) * c_scale
    elif search_scale:
        qbsum = 0
        for ci in range(0, C):
            for bi in range(0, B):
                for j in range(n_bit):
                    qbsum += vars01[ci * B * n_bit + bi * n_bit + j] * (j + low_bit) * sen_splits_list[bi]
        qbsum += equal_bit_val * sum(imp_splits_list) * c_rf
    else:
        raise NotImplementedError
        

    return qbsum # bit

def search_one_round(
    A, 
    imp_splits_list, 
    sen_splits_list,
    low_bit,
    sz_limit_bits,
    equal_bit_val=6,
    search_rf=True,
    search_scale=True,
    c_rf=7,
    c_scale=3,
    init_qbits=None
):
    C, B, n_bit = A.shape
    num_variable = C * B * n_bit
    A_flat = A.flatten().cpu().tolist()
    variable = {}
    for i in range(num_variable):
        variable[f"x{i}"] = LpVariable(f"x{i}", 0, 1, cat=LpInteger)
    
    # Warm start: 如果提供了上一次的 qbits 结果，设置初始值
    use_warm_start = False
    if init_qbits is not None:
        try:
            # init_qbits 是 [C_search, B] 的 qbits 数组（只包含搜索的通道）
            # 需要将其转换为 0-1 决策变量的初始值
            for ci in range(C):
                for bi in range(B):
                    for j in range(n_bit):
                        idx = ci * B * n_bit + bi * n_bit + j
                        if idx < num_variable:
                            # 如果这个 bit 等级匹配 init_qbits 中的值，设为 1，否则设为 0
                            if ci < init_qbits.shape[0] and bi < init_qbits.shape[1]:
                                expected_bit = j + low_bit
                                if int(init_qbits[ci][bi]) == expected_bit:
                                    variable[f"x{idx}"].setInitialValue(1)
                                else:
                                    variable[f"x{idx}"].setInitialValue(0)
            use_warm_start = True
            print(f"Warm start enabled with init_qbits shape {init_qbits.shape}")
        except Exception as e:
            print(f"Warm start failed: {e}, falling back to cold start")
            use_warm_start = False

    prob = LpProblem("Model_Size", LpMinimize)
    prob += qbit_channel_size_estimator(
        [variable[f"x{i}"] for i in range(num_variable)], 
        n_bit, 
        low_bit, 
        C, 
        B, 
        imp_splits_list, 
        sen_splits_list,
        equal_bit_val=equal_bit_val,
        search_rf=search_rf,
        search_scale=search_scale,
        c_rf=c_rf,
        c_scale=c_scale) <= sz_limit_bits
    
    for ci in range(0, C):
        for bi in range(0, B):
            le = ci * B * n_bit + bi * n_bit
            ri = le + n_bit 
            if ri <= num_variable:
                prob += sum([variable[f"x{i}"] for i in range(le, ri)]) == 1

    prob += sum([(variable[f"x{i}"]) * A_flat[i] for i in range(num_variable)])
    
    prob.solve(PULP_CBC_CMD(timeLimit=50, msg=True, warmStart=use_warm_start))

    # 获取 objective value
    obj_value = prob.objective.value() if prob.objective is not None else float('inf')
    print(f"Objective value: {obj_value}")

    solution = {}
    for i in range(num_variable):
        solution[f"x{i}"] = variable[f"x{i}"].varValue

    ret_vals = []
    qbits = np.zeros([C, B], dtype=int)
    for ci in range(C):
        for bi in range(0, B):
            for j in range(n_bit):
                idx = ci * B * n_bit + bi * n_bit + j
                ret_vals.append(int(variable[f"x{idx}"].varValue))
                if variable[f"x{idx}"].varValue != 0:
                    qbits[ci][bi] = j + low_bit
    
    if search_rf and (not search_scale):
        qbits = np.concatenate([qbits, np.ones([c_scale, B])*equal_bit_val], axis=0)
    if (not search_rf) and search_scale:
        qbits = np.concatenate([np.ones([c_rf, B])*equal_bit_val, qbits], axis=0)
    print('qbits in one round', qbits)
    return qbits, obj_value

def final_qbit_size_estimator(qbits: np.ndarray, splits: np.ndarray, n_channel, n_block):
    qbsz = 0
    for i in range(n_channel):
        for j in range(n_block):
            qbsz += qbits[i, j] * splits[i, j]
    return qbsz

def save_npz(
    oct,
    oct_param,
    ff,
    rf, 
    scales, 
    qbits: np.ndarray, 
    ntk,
    cb,
    imp_splits,
    sen_splits,
    n_block,
    depth,
    n_rf_channels=7,
    num_keep=0,
    cb_quant_bits=8,
    save_dir=None,
    cached_oct_bytes=None,
    cached_ntk_bytes=None
):
    trans = [depth, n_block]
    channel_check = qbits <= 8
    if save_dir is None:
        save_dir = tempfile.mkdtemp(prefix='compressgs_tmp_')
    else:
        shutil.rmtree(save_dir, ignore_errors=True)
        os.makedirs(save_dir, exist_ok=True)
    bin_dir = os.path.join(save_dir, 'bins')
    os.makedirs(bin_dir, exist_ok=True)
    
    # 使用 GPCC 编码 octree（使用缓存避免重复编码）
    if cached_oct_bytes is not None:
        if is_gpcc_available():
            with open(os.path.join(bin_dir, 'oct.gpcc'), 'wb') as f_oct:
                f_oct.write(cached_oct_bytes)
        else:
            with open(os.path.join(bin_dir, 'oct.npz'), 'wb') as f_oct:
                f_oct.write(cached_oct_bytes)
    elif is_gpcc_available():
        oct_bitstream = gpcc_encode_octree(oct, oct_param, depth)
        with open(os.path.join(bin_dir, 'oct.gpcc'), 'wb') as f_oct:
            f_oct.write(oct_bitstream)
    else:
        np.savez_compressed(os.path.join(bin_dir, 'oct'), points=oct, params=oct_param)
    
    # 使用 Laplace 范围编码 VQ indices（使用缓存避免重复编码）
    if cached_ntk_bytes is not None:
        with open(os.path.join(bin_dir, 'ntk.bin'), 'wb') as f_ntk:
            f_ntk.write(cached_ntk_bytes)
    else:
        ntk_bitstream = ntk_encode(ntk, n_block=n_block)
        with open(os.path.join(bin_dir, 'ntk.bin'), 'wb') as f_ntk:
            f_ntk.write(ntk_bitstream)
    
    # 如果有 TopK 量化，对整个 codebook 进行量化
    if num_keep > 0:
        from .meson_utils import quantize_kept_sh
        cb_torch = torch.from_numpy(cb).cuda() if isinstance(cb, np.ndarray) else cb
        cb_q_indices, cb_scales, cb_zero_points, cb_split = quantize_kept_sh(
            cb_torch,
            n_block=n_block,
            num_bits=cb_quant_bits
        )
        
        cb_q_np = cb_q_indices.detach().cpu().numpy()
        cb_scales_np = cb_scales.detach().cpu().numpy()
        cb_zps_np = cb_zero_points.detach().cpu().numpy()
        
        # Laplace 编码 codebook 量化索引
        cb_q_bitstream = laplace_encode_blocks(cb_q_np, cb_split, cb_q_np.shape[0])
        with open(os.path.join(bin_dir, 'cb_q.bin'), 'wb') as f_cb:
            f_cb.write(cb_q_bitstream)
        
        np.savez_compressed(
            os.path.join(bin_dir, 'cb_meta.npz'),
            scales=cb_scales_np,
            zero_points=cb_zps_np,
            split=np.array(cb_split),
            num_entries=cb.shape[0],
            num_keep=num_keep
        )
        
        cb_bytes = os.path.getsize(os.path.join(bin_dir, 'cb_q.bin')) + \
                   os.path.getsize(os.path.join(bin_dir, 'cb_meta.npz'))
    else:
        np.savez_compressed(os.path.join(bin_dir , 'um.npz'), umap=cb)
        cb_bytes = os.path.getsize(os.path.join(bin_dir , 'um.npz'))
    
    # 计算固定开销大小
    oct_file = os.path.join(bin_dir, 'oct.gpcc') if os.path.exists(os.path.join(bin_dir, 'oct.gpcc')) else os.path.join(bin_dir, 'oct.npz')
    ntk_file = os.path.join(bin_dir, 'ntk.bin')
    other_bytes = os.path.getsize(oct_file) + os.path.getsize(ntk_file) + cb_bytes
    
    # 量化 rf (RAHT AC 系数) 并用 Laplace 编码
    power_qbits_rf = (2 ** torch.tensor(qbits[:n_rf_channels, :], device='cuda') - 1).int()
    splits_tensor_rf = torch.tensor(imp_splits, device='cuda', dtype=torch.int32).unsqueeze(0).repeat(rf.shape[1], 1)
    q_indices_rf, scales_rf, zps_rf, _, _ = pure_quant_wo_minmax(rf, splits_tensor_rf, power_qbits_rf)
    q_rf_np = q_indices_rf.cpu().numpy()  # [C_rf, N]
    
    rf_sz = torch.stack([scales_rf, zps_rf], dim=-1).flatten().cpu().tolist()
    trans.extend(rf_sz)
    
    # 量化 scales 并用 Laplace 编码
    power_qbits_sc = (2 ** torch.tensor(qbits[n_rf_channels:, :], device='cuda') - 1).int()
    splits_tensor_sc = torch.tensor(sen_splits, device='cuda', dtype=torch.int32).unsqueeze(0).repeat(scales.shape[1], 1)
    q_indices_sc, scales_sc, zps_sc, _, _ = pure_quant_wo_minmax(scales, splits_tensor_sc, power_qbits_sc)
    q_sc_np = q_indices_sc.cpu().numpy()  # [C_sc, N]
    
    sc_sz = torch.stack([scales_sc, zps_sc], dim=-1).flatten().cpu().tolist()
    trans.extend(sc_sz)
    
    # Laplace 编码 rf 和 scales
    rf_bitstream = laplace_encode_blocks(q_rf_np, imp_splits, rf.shape[1])
    with open(os.path.join(bin_dir, 'orgb.bin'), 'wb') as f_orgb:
        f_orgb.write(rf_bitstream)
    np.savez_compressed(os.path.join(bin_dir, 'orgb_dc.npz'), f=ff.cpu().numpy())
    
    ct_bitstream = laplace_encode_blocks(q_sc_np, sen_splits, scales.shape[1])
    with open(os.path.join(bin_dir, 'ct.bin'), 'wb') as f_ct:
        f_ct.write(ct_bitstream)
    
    np.savez_compressed(os.path.join(bin_dir, 't.npz'), t=np.array(trans))
    
    bin_zip_path = os.path.join(save_dir, 'bins.zip')
    os.system(f'zip -jq {bin_zip_path} {bin_dir}/*')
    zip_file_bytes = os.path.getsize(bin_zip_path)
    # 清理临时目录
    shutil.rmtree(save_dir, ignore_errors=True)
    return zip_file_bytes, other_bytes
    
def search_qbits(
    n_round,
    depth,
    n_block,
    oct,
    oct_param,
    fdc,
    opa,
    scales,
    r,
    ntk,
    cb,
    low_bit,
    high_bit,
    size_limit_mb,
    search_rf=True,
    search_scale=True,
    fluc_percent=0.01,
    dis_type=2,
    equal_bit_val=6,
    use_quat=False,
    num_keep=0,
    cb_quant_bits=8,
    init_qbits=None,
    use_raht=True,
    channel_importance_weight=False,
    percentile_quant=False,
    auto_entropy_model=False
):
    _, V = decode_oct(oct_param, oct, depth)
    w, val, reorder = copyAsort(V)
    res = haar3D_param(depth, w, val)
    res_inv = inv_haar3D_param(V, depth)
    
    # @junchen: rm eulers. see. --use_quat
    if use_quat:
        f1 = torch.concat([opa, r, fdc.contiguous().squeeze()], -1)
        n_rf_channels = 8
    else:
        norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
        q = r / norm[:, None]
        eulers = ToEulerAngles_FT(q, save=False)
        f1 = torch.concat([opa, eulers, fdc.contiguous().squeeze()], -1)
        n_rf_channels = 7
    
    C = f1[reorder].cuda()
    if use_raht:
        iW1 = res['iW1']
        iW2 = res['iW2']
        iLeft_idx = res['iLeft_idx']
        iRight_idx = res['iRight_idx']
        
        for d in range(depth * 3):
            w1 = iW1[d]
            w2 = iW2[d]
            left_idx = iLeft_idx[d]
            right_idx = iRight_idx[d]
            C[left_idx], C[right_idx] = transform_batched_torch(
                w1, 
                w2, 
                C[left_idx], 
                C[right_idx]
            )
    else:
        print("RAHT ablation: skipping RAHT forward transform in search_qbits")
    rf = C[1:].clone().detach().cuda()
    # print('dis_type', dis_type)
    
    # === Improvement: Percentile quantization (clip outliers) ===
    # Clip extreme outliers to reduce quantization range, giving more precision to the bulk of data.
    # Using 0.1st/99.9th percentile to be conservative (only clip extreme 0.1% outliers each side).
    if percentile_quant:
        PLOW, PHIGH = 0.001, 0.999  # 0.1st / 99.9th percentile
        print(f"Improvement: Applying percentile quantization (clip to [{PLOW*100:.1f}th, {PHIGH*100:.1f}th] percentile)")
        for c in range(rf.shape[1]):
            col = rf[:, c]
            p_lo = torch.quantile(col.float(), PLOW)
            p_hi = torch.quantile(col.float(), PHIGH)
            rf[:, c] = col.clamp(p_lo, p_hi)
        for c in range(scales.shape[1]):
            col = scales[:, c]
            p_lo = torch.quantile(col.float(), PLOW)
            p_hi = torch.quantile(col.float(), PHIGH)
            scales[:, c] = col.clamp(p_lo, p_hi)
    
    #预计算率失真（Rate-Distortion）代价矩阵，为后续的 0-1 整数规划（ILP）求解器提供优化的目标函数系数。
    if search_rf and search_scale:
        A_rf, B_rf, _ = get_ddrf(
            rf, n_block, n_channel=n_rf_channels, dis_type=dis_type, low_bit=low_bit, high_bit=high_bit
        )
        A_scale, B_scale, _ = get_ddrf(
            scales, n_block, n_channel=3, dis_type=dis_type, low_bit=low_bit, high_bit=high_bit
        )
        # === Improvement: Channel importance weight ===
        if channel_importance_weight:
            print("Improvement: Applying channel importance weights to ILP objective")
            # use_quat=True: channels are [opacity(1), quat(4), features_dc(3)] = 8 channels
            # use_quat=False: channels are [opacity(1), euler(3), features_dc(3)] = 7 channels
            if use_quat:
                # Channel weights: opacity=2.0, quat(4)=1.5, features_dc(3)=1.0
                rf_weights = torch.tensor([2.0, 1.5, 1.5, 1.5, 1.5, 1.0, 1.0, 1.0], device=A_rf.device)
            else:
                # Channel weights: opacity=2.0, euler(3)=1.5, features_dc(3)=1.0
                rf_weights = torch.tensor([2.0, 1.5, 1.5, 1.5, 1.0, 1.0, 1.0], device=A_rf.device)
            # Scaling weights: 0.8 for all 3 channels
            scale_weights = torch.tensor([0.8, 0.8, 0.8], device=A_scale.device)
            
            # Apply weights: A_rf is [C_rf, B, n_bit], multiply distortion by weight
            A_rf = A_rf * rf_weights[:, None, None]
            A_scale = A_scale * scale_weights[:, None, None]
            print(f"  rf_weights: {rf_weights.tolist()}, scale_weights: {scale_weights.tolist()}")
        
        A = torch.concat([A_rf, A_scale], dim=0) # give rf a weight as it is more fragile.
    elif search_rf:
        A_rf, B_rf, _ = get_ddrf(
            rf, n_block, n_channel=n_rf_channels, dis_type=dis_type, low_bit=low_bit, high_bit=high_bit
        )
        print('not search scale')
        A = A_rf # rgb,...
    elif search_scale:
        A_scale, B_scale, _ = get_ddrf(
            scales, n_block, n_channel=3, dis_type=dis_type, low_bit=low_bit, high_bit=high_bit
        )
        A = A_scale
    
    n_c, n_b, n_bit = A.shape
    num_variable = n_c * n_b * n_bit
    # A_flat = A.flatten().cpu().tolist()
    imp_splits_list = split_length(rf.shape[0] - 1, n_block)
    sen_splits_list = split_length(scales.shape[0], n_block)
    c_rf = rf.shape[1]
    c_scale = scales.shape[1]
    model_size_limit = size_limit_mb * 1024 * 1024 * 8
    full_splits = torch.concat([
        torch.tensor(imp_splits_list).reshape(1, -1).repeat(n_rf_channels, 1), 
        torch.tensor(sen_splits_list).reshape(1, -1).repeat(3, 1)
    ], dim = 0)
    full_splits = full_splits.numpy()
    actual_sizes = []
    qbit_size_limits = []
    
    # 预先编码 oct 和 ntk，缓存结果供后续 save_npz 调用使用（避免重复编码）
    import io
    if is_gpcc_available():
        print("Pre-encoding octree with GPCC...", flush=True)
        cached_oct_bytes = gpcc_encode_octree(oct, oct_param, depth)
        print(f"  GPCC oct: {len(cached_oct_bytes)} bytes ({len(cached_oct_bytes)/1024/1024:.2f} MB)")
    else:
        buf = io.BytesIO()
        np.savez_compressed(buf, points=oct, params=oct_param)
        cached_oct_bytes = buf.getvalue()
        print(f"  zlib oct: {len(cached_oct_bytes)} bytes ({len(cached_oct_bytes)/1024/1024:.2f} MB)")
    
    cached_ntk_bytes = ntk_encode(ntk, n_block=n_block)
    print(f"  Laplace ntk: {len(cached_ntk_bytes)} bytes ({len(cached_ntk_bytes)/1024/1024:.2f} MB)")
    
    qbits_8 = np.ones([n_rf_channels + 3, A.shape[1]], dtype=int) * 8
    save_size_bytes, other_bytes = save_npz(
        oct,
        oct_param,
        C[0],
        rf,
        scales,
        qbits_8,
        ntk,
        cb,
        imp_splits_list,
        sen_splits_list,
        n_block,
        depth,
        n_rf_channels=n_rf_channels,
        num_keep=num_keep,
        cb_quant_bits=cb_quant_bits,
        cached_oct_bytes=cached_oct_bytes,
        cached_ntk_bytes=cached_ntk_bytes
    )
    print(f"initial fe: {save_size_bytes / 1024/1024} MB {save_size_bytes} B")
    print(f"initial other: {other_bytes / 1024/1024} MB {other_bytes} B")
    delta_bits = model_size_limit - save_size_bytes * 8 
    qbit_size_limit = 8 * rf.shape[0] * (n_rf_channels + 3) + delta_bits

    # P0: 检测负预算 — codebook+其他开销已超过 size_limit，ILP 无法求解
    if qbit_size_limit <= 0:
        print(f"[ERROR] Negative qbit_size_limit={qbit_size_limit/8/1024/1024:.4f}MB! "
              f"Overhead ({save_size_bytes/1024/1024:.2f}MB) exceeds size_limit ({size_limit_mb:.2f}MB). "
              f"Returning minimum qbits (all low_bit={low_bit}).")
        qbits_min = np.ones([n_rf_channels + 3, A.shape[1]], dtype=int) * low_bit
        return qbits_min, float('inf')
    # 第一轮搜索时使用 init_qbits（如果提供），后续轮次复用上一轮结果
    current_init_qbits = None
    if init_qbits is not None:
        # init_qbits 是完整的 [C_total, B] qbits 数组，需要提取搜索部分
        if search_rf and search_scale:
            current_init_qbits = init_qbits[:n_rf_channels + 3]
        elif search_rf:
            current_init_qbits = init_qbits[:n_rf_channels]
        elif search_scale:
            current_init_qbits = init_qbits[n_rf_channels:]
        print(f"Using warm start init_qbits with shape {current_init_qbits.shape}")
    
    obj_value = float('inf')
    for ri in range(n_round):
        qbits, obj_value = search_one_round(
            A,
            imp_splits_list=imp_splits_list,
            sen_splits_list=sen_splits_list,
            low_bit=low_bit,
            sz_limit_bits=qbit_size_limit,
            equal_bit_val=equal_bit_val,
            search_rf=search_rf,
            search_scale=search_scale,
            c_rf=c_rf,
            c_scale=c_scale,
            init_qbits=current_init_qbits
        )
        
        # 后续轮次用本轮结果作为 warm start
        if search_rf and search_scale:
            current_init_qbits = qbits[:n_rf_channels + 3]
        elif search_rf:
            current_init_qbits = qbits[:n_rf_channels]
        elif search_scale:
            current_init_qbits = qbits[n_rf_channels:]
        else:
            current_init_qbits = qbits
        
        save_size_bytes, other_bytes = save_npz(
            oct,
            oct_param,
            C[0],
            rf,
            scales,
            qbits,
            ntk,
            cb,
            imp_splits_list,
            sen_splits_list,
            n_block,
            depth,
            n_rf_channels=n_rf_channels,
            num_keep=num_keep,
            cb_quant_bits=cb_quant_bits,
            cached_oct_bytes=cached_oct_bytes,
            cached_ntk_bytes=cached_ntk_bytes
        )
        actual_size = save_size_bytes * 8
        if ri > 0:
            delta_actual_size = actual_size - actual_sizes[-1] + 0.1
            delta_qbit_limit = qbit_size_limit - qbit_size_limits[-1] + 0.1
        else:
            delta_actual_size = 1
            delta_qbit_limit = 1
        actual_sizes.append(actual_size)
        qbit_size_limits.append(qbit_size_limit)
        searched_qbit_size = final_qbit_size_estimator(
            qbits,
            full_splits,
            n_channel=n_rf_channels + 3,
            n_block=n_block
        )
        print(f"{ri} old size limit: {qbit_size_limit / 8 / 1024/1024} MB {qbit_size_limit / 8} B")
        print(f"{ri} searched_qbit_size: {searched_qbit_size / 8 / 1024/1024} MB {searched_qbit_size / 8} B")
        delta_qbit = delta_qbit_limit / delta_actual_size * (model_size_limit - actual_size)
        if delta_qbit > actual_size:
            delta_qbit = model_size_limit - actual_size
        qbit_size_limit = qbit_size_limit + delta_qbit
        print(f"{ri} actual full size: {actual_size / 8 / 1024/1024} MB {actual_size / 8} B")
        print(f"{ri} new size limit: {qbit_size_limit / 8 / 1024/1024} MB {qbit_size_limit / 8} B")
        if abs(actual_size - model_size_limit) / model_size_limit < fluc_percent:
            break            
    
    return qbits, obj_value
    