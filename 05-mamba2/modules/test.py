import torch
# 1. 直接导入官方安装的 Mamba2 模块
import mamba2

def debug_run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    # 2. 初始化官方模型
    # 核心关键点：use_mem_eff_path=False
    # 这会强制官方源码走 else 分支，而不是去跑黑盒 Kernel
    model = mamba2.Mamba2Simple(
        d_model=256,
        use_mem_eff_path=False 
    ).to(device)

    # 3. 造数据
    # batch=1, seq_len=128, d_model=256
    x = torch.linspace(0, 1, 256).to(device)
    x = x.unsqueeze(0).unsqueeze(0).repeat(1, 128, 1)
    print(x.shape)

    # x = torch.randn(1, 128, 256).to(device)

    print("🚀 开始运行... 请确保你已经在官方源码文件里打好了断点！")
    y = model(x) 
    print(y[0, :5, :10])

if __name__ == "__main__":
    debug_run()