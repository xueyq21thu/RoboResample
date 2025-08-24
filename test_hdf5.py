import h5py

# 把你的文件路径放在这里
file_path = '/mnt/disk_2/guanxing/xueyq21thu/BC-IB/rollout_succ/libero_spatial_succ/libero_spatial_pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5'

# def print_hdf5_structure(name, obj):
#     print(name)

# with h5py.File(file_path, 'r') as f:
#     print(f"--- Structure of {file_path} ---")
#     f.visititems(print_hdf5_structure)
    
#     # 更具体地检查第一个任务的观测键
#     print("\n--- Keys available in task 0's observations ---")
#     try:
#         task_0_obs_keys = list(f['data/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket/obs'].keys())
#         print(task_0_obs_keys)
#     except KeyError as e:
#         print(f"Could not access task 0 obs. Error: {e}")
f = h5py.File(file_path, 'r')
print(list(f["data/demo_0/obs"].keys()))
