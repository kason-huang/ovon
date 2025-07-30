import subprocess

def main():
    try:
        # 定义命令参数（避免使用 shell=True 防止安全风险）
        command = [
            "python", "-m", "ovon.run",
            "--run-type", "train",
            "--exp-config", "config/experiments/transformer_dagger.yaml"
        ]
        
        # 执行命令并捕获输出
        result = subprocess.run(
            command,
            check=True,          # 检查命令是否成功（非0返回码时抛出异常）
            text=True,           # 输出以文本形式返回
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 打印输出结果
        print("标准输出:\n", result.stdout)
        print("执行成功！")
    
    except subprocess.CalledProcessError as e:
        # 处理命令执行失败的情况
        print(f"命令执行失败（返回码 {e.returncode}):")
        print("标准错误:\n", e.stderr)
    except FileNotFoundError:
        print("错误：未找到 'python' 解释器或 'ovon.run' 模块。")

if __name__ == "__main__":
    main()