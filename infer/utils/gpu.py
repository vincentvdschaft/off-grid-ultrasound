import subprocess

def get_available_vram():
    """Returns a list of available VRAM in MiB for each GPU"""
    
    # Execute nvidia-smi and capture the output
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], 
                            stdout=subprocess.PIPE)
    # Decode the output and split into lines
    vram_free = result.stdout.decode('utf-8').strip().split('\n')
    # Convert the output to a list of integers representing free memory in MiB
    vram_free = [int(x) for x in vram_free]
    
    return vram_free

def floor_to(value, step=128):
    """Floors the value to the nearest multiple of step"""
    return value - (value % step)