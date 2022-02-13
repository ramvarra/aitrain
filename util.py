
def show_tf_info(tf):
    '''
    Display information about the TF package
    '''
    print('TF Version:', tf.__version__)
    print('Logical Devices:', tf.config.list_logical_devices())
    print('Physical Devices:', tf.config.list_physical_devices())
    