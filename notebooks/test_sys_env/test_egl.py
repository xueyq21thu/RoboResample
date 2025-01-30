import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['EGL_PLATFORM'] = 'surfaceless'
os.environ['DISPLAY'] = ''
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

try:
    import OpenGL.EGL as egl
    print("EGL库已成功导入")
    
    # 初始化EGL
    display = egl.eglGetDisplay(egl.EGL_DEFAULT_DISPLAY)
    if display == egl.EGL_NO_DISPLAY:
        print("无法获取EGL显示")
        exit(1)
    
    # 初始化EGL
    major, minor = egl.EGLint(), egl.EGLint()
    if not egl.eglInitialize(display, major, minor):
        print("EGL初始化失败")
        exit(1)
    
    print(f"EGL版本: {major.value}.{minor.value}")
    
    # 获取支持的客户端API
    apis = egl.eglQueryString(display, egl.EGL_CLIENT_APIS)
    print("支持的客户端API:", apis)
    
    # 关闭EGL
    egl.eglTerminate(display)
    
except ImportError:
    print("无法导入EGL库")
except Exception as e:
    print(f"发生错误: {e}")