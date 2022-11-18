def generate_sample_output(raw, rgb_map, disp_map, acc_map, weights, depth_map):
    return {
        'raw': raw,
        'rgb_map': rgb_map,
        'disp_map': disp_map,
        'acc_map': acc_map,
        'weights': weights,
        'depth_map': depth_map,
    }