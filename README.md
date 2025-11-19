# PDI-PIA

# Instalación de CuPy con soporte para CUDA

Este proyecto utiliza **CuPy** para acelerar las operaciones numéricas utilizando la GPU. Por defecto, la instalación traerá soporte para **CUDA 12**, pero si necesitas usar una versión diferente de **CUDA**, puedes instalar la versión que más te convenga.

Si **CuPy** no está disponible en tu sistema (por ejemplo, si no tienes soporte para GPU o CUDA), el proyecto automáticamente usará **NumPy** como alternativa. Esto se maneja a través de un bloque de código que intenta importar **CuPy**, y si falla, recurre a **NumPy** sin necesidad de intervención del usuario.

Si deseas usar una versión diferente de **CUDA** (por ejemplo, **CUDA 11.x** o **CUDA 13.x**), sigue estos pasos:

1. **Desinstalar la versión actual de CuPy con CUDA 12**:
   Si ya tienes instalada la versión de **CUDA 12**, primero desinstálala:

   ```bash
   pip uninstall cupy-cuda12x
   ```

2. **Instalar la versión deseada de CuPy**:
   Luego, instala la versión de **CuPy** que corresponda con la versión de **CUDA** que tienes instalada:

   * Para **CUDA 11.x**:

     ```bash
     pip install cupy-cuda11x  # O cupy-cuda111 o cupy-cuda112 dependiendo de tu versión exacta de CUDA 11
     ```

   * Para **CUDA 13.x**:

     ```bash
     pip install cupy-cuda13x  # Para CuPy con soporte para CUDA 13
     ```

## Sin GPU o sin CUDA

Si no tienes una GPU o prefieres usar la CPU en lugar de CUDA, puedes instalar CuPy con soporte solo para CPU:

```bash
pip install cupy-cuda-none
```

Esta versión no requiere CUDA y funcionará exclusivamente en la CPU.
