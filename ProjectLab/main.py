import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# ============================================================
# Параметры задачи
# ============================================================
L = 0.1
T_star = 100.0

# Граничные условия (в относительных единицах T / T*)
T_AB = 1.0   # левая стена   x = 0
T_BC = 2.0   # верхняя стена y = L
T_CD = 3.0   # правая стена  x = L
T_DA = 4.0   # нижняя стена  y = 0

N = 41
h = L / (N - 1)

tol = 1e-8
max_iter = 100_000

omega_opt = 2.0 / (1.0 + np.sin(np.pi * h / L))
print(f"Оптимальный omega (SOR) = {omega_opt:.6f}")


def make_initial(N):
    u = np.zeros((N, N))
    # u[l, m]:  l — индекс по y (строка), m — индекс по x (столбец)
    u[0, :]  = T_DA       # нижняя стена  y=0
    u[-1, :] = T_BC       # верхняя стена y=L
    u[:, 0]  = T_AB       # левая стена   x=0
    u[:, -1] = T_CD       # правая стена  x=L
    # углы — среднее двух примыкающих стен
    u[0, 0]   = 0.5 * (T_DA + T_AB)
    u[-1, 0]  = 0.5 * (T_BC + T_AB)
    u[-1, -1] = 0.5 * (T_BC + T_CD)
    u[0, -1]  = 0.5 * (T_DA + T_CD)
    return u


# ============================================================
# Метод Якоби — «Командный честный шаг»
# ============================================================
# Суть: ВСЕ 4 соседа берутся ТОЛЬКО со старой итерации (i).
# Новые значения пишутся в «черновик» (u_new), который
# становится основной картой только после полного прохода.
#
# Формула:
#   u_{m,l}^{i+1} = ( u_{m-1,l}^{i} + u_{m+1,l}^{i}
#                    + u_{m,l-1}^{i} + u_{m,l+1}^{i} ) / 4
#
# Плюс: полностью параллелизуем (каждый узел независим).
# Минус: медленный — тепло ползёт на 1 узел за 1 итерацию,
#        потому что обновлённые значения НЕ используются
#        до следующего шага.
# ============================================================
def jacobi(N, tol, max_iter):
    u = make_initial(N)
    residuals = []
    for it in range(1, max_iter + 1):
        u_old = u.copy()  # «чистовик» прошлого шага — читаем ТОЛЬКО отсюда
        # Все 4 соседа — со старой итерации u_old
        u[1:-1, 1:-1] = 0.25 * (
            u_old[1:-1, 0:-2] +   # сосед слева   u_{m-1,l}^{i}
            u_old[1:-1, 2:]   +   # сосед справа  u_{m+1,l}^{i}
            u_old[0:-2, 1:-1] +   # сосед снизу   u_{m,l-1}^{i}
            u_old[2:,   1:-1]     # сосед сверху  u_{m,l+1}^{i}
        )
        diff = np.max(np.abs(u[1:-1, 1:-1] - u_old[1:-1, 1:-1]))
        residuals.append(diff)
        if diff < tol:
            print(f"Якоби сошёлся за {it} итераций")
            break
    else:
        print(f"Якоби: достигнут лимит {max_iter} итераций")
    return u, residuals


# ============================================================
# Метод Зейделя — «Жадный эгоистичный шаг»
# ============================================================
# Суть: идём по сетке по порядку (l↑, m→). Как только
# посчитали новое значение узла — СРАЗУ записываем в чистовик.
# Следующий сосед уже видит обновлённые значения.
#
# При обходе l = 1→N-2, m = 1→N-2:
#   u_{m-1,l} — сосед слева  — УЖЕ обновлён (i+1)
#   u_{m+1,l} — сосед справа — ЕЩЁ старый   (i)
#   u_{m,l-1} — сосед снизу  — УЖЕ обновлён (i+1)
#   u_{m,l+1} — сосед сверху — ЕЩЁ старый   (i)
#
# Формула:
#   u_{m,l}^{i+1} = ( u_{m-1,l}^{i+1} + u_{m+1,l}^{i}
#                    + u_{m,l-1}^{i+1} + u_{m,l+1}^{i} ) / 4
#
# Плюс: тепло «бежит» в 2 раза быстрее → сходимость ~2× Якоби.
# Минус: нельзя параллелить — порядок обхода важен.
# ============================================================
def gauss_seidel(N, tol, max_iter):
    u = make_initial(N)
    residuals = []
    for it in range(1, max_iter + 1):
        max_diff = 0.0
        for l in range(1, N - 1):        # снизу вверх
            for m in range(1, N - 1):    # слева направо
                u_new = 0.25 * (
                    u[l, m - 1] +    # слева  — уже обновлён (i+1)
                    u[l, m + 1] +    # справа — ещё старый   (i)
                    u[l - 1, m] +    # снизу  — уже обновлён (i+1)
                    u[l + 1, m]      # сверху — ещё старый   (i)
                )
                max_diff = max(max_diff, abs(u_new - u[l, m]))
                u[l, m] = u_new      # сразу в чистовик!
        residuals.append(max_diff)
        if max_diff < tol:
            print(f"Зейдель сошёлся за {it} итераций")
            break
    else:
        print(f"Зейдель: достигнут лимит {max_iter} итераций")
    return u, residuals


# ============================================================
# SOR — «Зейдель с разгоном»
# ============================================================
# Суть: берём шаг Зейделя, но «перелетаем» через него
# с коэффициентом ω > 1. Как будто мы знаем, что Зейдель
# двигается в правильном направлении, и делаем шаг побольше.
#
# Формула:
#   u_gs = (среднее 4 соседей, как в Зейделе)
#   u_{m,l}^{i+1} = (1 - ω) * u_{m,l}^{i} + ω * u_gs
#
# При ω = 1 → чистый Зейдель.
# При ω_opt ≈ 2/(1+sin(πh/L)) → максимальное ускорение.
# При ω ≥ 2 → расходится.
# ============================================================
def sor(N, omega, tol, max_iter, verbose=True):
    u = make_initial(N)
    residuals = []
    for it in range(1, max_iter + 1):
        max_diff = 0.0
        for l in range(1, N - 1):
            for m in range(1, N - 1):
                u_gs = 0.25 * (
                    u[l, m - 1] +    # слева  (i+1)
                    u[l, m + 1] +    # справа (i)
                    u[l - 1, m] +    # снизу  (i+1)
                    u[l + 1, m]      # сверху (i)
                )
                # Разгон: шагаем дальше, чем Зейдель
                u_new = (1.0 - omega) * u[l, m] + omega * u_gs
                max_diff = max(max_diff, abs(u_new - u[l, m]))
                u[l, m] = u_new
        residuals.append(max_diff)
        if max_diff < tol:
            if verbose:
                print(f"SOR (ω={omega:.4f}) сошёлся за {it} итераций")
            break
    else:
        if verbose:
            print(f"SOR: достигнут лимит {max_iter} итераций")
    return u, residuals


# ============================================================
# Решение
# ============================================================
print("=" * 55)
print("Решение уравнения Лапласа на квадрате")
print(f"Сетка {N}×{N}, h = {h:.6f} м")
print(f"T_AB={T_AB}, T_BC={T_BC}, T_CD={T_CD}, T_DA={T_DA} (×T*={T_star}°C)")
print("=" * 55)

u_jac, res_jac = jacobi(N, tol, max_iter)
u_gs,  res_gs  = gauss_seidel(N, tol, max_iter)
u_sor, res_sor = sor(N, omega_opt, tol, max_iter)

# Проверка: все методы сходятся к одному решению
print(f"\nmax |u_jacobi - u_seidel| = {np.max(np.abs(u_jac - u_gs)):.2e}")
print(f"max |u_seidel - u_sor|    = {np.max(np.abs(u_gs - u_sor)):.2e}")

# ============================================================
# Исследование SOR от omega
# ============================================================
print("\nSOR: зависимость от ω")
omegas_test = np.arange(1.0, 1.99, 0.1)
iters_list = []
for w in omegas_test:
    _, res = sor(N, w, tol, max_iter, verbose=False)
    n_it = len(res)
    iters_list.append(n_it)
    print(f"  ω = {w:.2f} -> {n_it} итераций")

# ============================================================
# ГРАФИКИ
# ============================================================
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

# --- 1. Сходимость всех методов (лог-шкала) ---
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.semilogy(res_jac, label=f'Якоби ({len(res_jac)} ит.)', linewidth=1.5)
ax1.semilogy(res_gs,  label=f'Зейдель ({len(res_gs)} ит.)', linewidth=1.5)
ax1.semilogy(res_sor, label=f'SOR ω={omega_opt:.3f} ({len(res_sor)} ит.)',
             linewidth=1.5)
ax1.set_xlabel('Номер итерации')
ax1.set_ylabel(r'$\max |u^{k+1} - u^{k}|$')
ax1.set_title('Сравнение скорости сходимости')
ax1.legend(fontsize=11)
ax1.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig('convergence.png', dpi=150)
plt.show()

# --- 2. Изолинии T(x,y) ---
fig2, ax2 = plt.subplots(figsize=(8, 7))
T_min = min(T_AB, T_BC, T_CD, T_DA) * T_star
T_max = max(T_AB, T_BC, T_CD, T_DA) * T_star
levels = np.linspace(T_min, T_max, 25)
cf = ax2.contourf(X * 1000, Y * 1000, u_sor * T_star, levels=levels, cmap='hot')
cs = ax2.contour(X * 1000, Y * 1000, u_sor * T_star, levels=levels,
                 colors='k', linewidths=0.5)
ax2.clabel(cs, fontsize=7, fmt='%.0f')
plt.colorbar(cf, ax=ax2, label='T, °C')
ax2.set_xlabel('x, мм')
ax2.set_ylabel('y, мм')
ax2.set_title('Распределение температуры T(x, y)')
ax2.set_aspect('equal')
plt.tight_layout()
plt.savefig('temperature_field.png', dpi=150)
plt.show()

# --- 3. 3D-поверхность ---
fig3 = plt.figure(figsize=(10, 7))
ax3 = fig3.add_subplot(111, projection='3d')
surf = ax3.plot_surface(X * 1000, Y * 1000, u_sor * T_star,
                        cmap='hot', edgecolor='none', alpha=0.9)
ax3.set_xlabel('x, мм')
ax3.set_ylabel('y, мм')
ax3.set_zlabel('T, °C')
ax3.set_title('Распределение температуры (3D)')
fig3.colorbar(surf, ax=ax3, shrink=0.5, label='T, °C')
plt.tight_layout()
plt.savefig('temperature_3d.png', dpi=150)
plt.show()

# --- 4. Сечения T(x) и T(y) ---
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))
y_indices = [N // 4, N // 2, 3 * N // 4]
for idx in y_indices:
    ax4a.plot(x * 1000, u_sor[idx, :] * T_star,
              label=f'y = {y[idx]*1000:.1f} мм')
ax4a.set_xlabel('x, мм')
ax4a.set_ylabel('T, °C')
ax4a.set_title('T(x) при различных y')
ax4a.legend()
ax4a.grid(True, ls='--', alpha=0.5)

x_indices = [N // 4, N // 2, 3 * N // 4]
for idx in x_indices:
    ax4b.plot(y * 1000, u_sor[:, idx] * T_star,
              label=f'x = {x[idx]*1000:.1f} мм')
ax4b.set_xlabel('y, мм')
ax4b.set_ylabel('T, °C')
ax4b.set_title('T(y) при различных x')
ax4b.legend()
ax4b.grid(True, ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig('temperature_sections.png', dpi=150)
plt.show()

# --- 5. SOR: итерации от omega ---
fig5, ax5 = plt.subplots(figsize=(8, 5))
ax5.plot(omegas_test, iters_list, 'o-', markersize=6)
ax5.axvline(omega_opt, color='r', ls='--', label=f'ω_opt = {omega_opt:.3f}')
ax5.set_xlabel('ω')
ax5.set_ylabel('Число итераций')
ax5.set_title('SOR: зависимость числа итераций от ω')
ax5.legend()
ax5.grid(True, ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig('sor_omega.png', dpi=150)
plt.show()

print("\nГотово! Графики сохранены.")
