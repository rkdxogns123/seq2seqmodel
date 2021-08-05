import tensorflow as tf


def set_gpu(memory_limit: int, gpu_number: int = 0):
    """
    GPU를 설정합니다.
    Args:
        memory_limit (int): 사용할 GPU 메모리의 최댓값
        gpu_number: 사용할 GPU의 번호
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    tf.config.experimental.set_virtual_device_configuration(gpus[gpu_number], [
        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])