import numpy as np

# 5a + 65b + 1071c = 995
# 65a + 1071b + 20123c = 18741
# 1071a + 20123b + 406275c = 378417

A = np.array([
    [5, 65, 1071],
    [65, 1071, 20123],
    [1071, 20123, 406275]
])

B = np.array([995, 18741, 378417])

try:
    solucion = np.linalg.solve(A, B)
    a_estimado = solucion[0]
    b_estimado = solucion[1]
    c_estimado = solucion[2]

    print(f"Estimador de a: {a_estimado}")
    print(f"Estimador de b: {b_estimado}")
    print(f"Estimador de c: {c_estimado}")

    print("\nVerificación:")
    print(
        f"Ecuación 1: {A[0,0]*a_estimado + A[0,1]*b_estimado + A[0,2]*c_estimado} (Esperado: {B[0]})")
    print(
        f"Ecuación 2: {A[1,0]*a_estimado + A[1,1]*b_estimado + A[1,2]*c_estimado} (Esperado: {B[1]})")
    print(
        f"Ecuación 3: {A[2,0]*a_estimado + A[2,1]*b_estimado + A[2,2]*c_estimado} (Esperado: {B[2]})")

except np.linalg.LinAlgError as e:
    print(f"Error al resolver el sistema: {e}")
    print("La matriz de coeficientes podría ser singular o no invertible.")
