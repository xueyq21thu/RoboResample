import torch
from lightning.fabric import Fabric


def diagnose_fabric_setup():
    try:
        # 检查CUDA可用性
        if not torch.cuda.is_available():
            print("警告：CUDA不可用")
            return

        # 检查设备数量是否匹配
        cuda_device_count = torch.cuda.device_count()
        config_gpu_count = len([0])
        
        if config_gpu_count > cuda_device_count:
            print(f"警告：配置的GPU数量({config_gpu_count})超过系统可用GPU数量({cuda_device_count})")
            return

        # 尝试创建Fabric实例
        fabric = Fabric(
            accelerator="cuda", 
            devices=[0], 
            precision="bf16-mixed", 
            strategy="deepspeed"
        )
        print("Fabric实例创建成功")

    except Exception as e:
        print(f"创建Fabric实例时发生错误: {e}")
        import traceback
        traceback.print_exc()

# 调用诊断函数
diagnose_fabric_setup()