import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

np.set_printoptions(precision=10, suppress=True)

def generate_random_spd(n, seed=None):
    """СПО: A = Mᵀ M + n·I.  SOR сходится при 0 < ω < 2."""
    rng = np.random.default_rng(seed)
    M = rng.uniform(-10, 10, size=(n, n))
    A = M.T @ M + n * np.eye(n)
    x_true = rng.uniform(-5, 5, size=n)
    b = A @ x_true
    return A, b, x_true

def generate_random(n, seed=None):
    rng = np.random.default_rng(seed)
    A = rng.uniform(-10, 10, size=(n, n))
    while np.abs(np.linalg.det(A)) < 1e-6:
        A = rng.uniform(-10, 10, size=(n, n))
    
    x_true = rng.uniform(-5, 5, size=n)
    b = A @ x_true
    return A, b, x_true

def read_system_from_file(path="slau.txt"):
    with open(path) as f:
        n = int(f.readline())
        A = np.zeros((n, n))
        for i in range(n):
            A[i] = list(map(float, f.readline().split()))
        b = np.array(list(map(float, f.readline().split())))
    return A, b, None

def compute_optimal_omega(A):
    """
    T = I − D⁻¹A,  ρ = ρ(T).
    Если ρ < 1:  ω = 1 + (ρ / (1 + √(1 − ρ²)))²
    Иначе:       ω = 1.0 (fallback на Зейдель).
    """
    n = A.shape[0]
    diag = np.diag(A)

    if np.any(np.abs(diag) < 1e-15):
        return 1.0, None, False

    D_inv = np.diag(1.0 / diag)
    T = np.eye(n) - D_inv @ A
    rho = np.max(np.abs(np.linalg.eigvals(T)))

    if rho >= 1.0:
        return 1.0, rho, False

    omega = 1.0 + (rho / (1.0 + np.sqrt(1.0 - rho ** 2))) ** 2
    return omega, rho, True

def sor(A, b, omega, tol=1e-12, max_iter=10000):
    DIVERGE_THRESHOLD = 1e15

    n = A.shape[0]
    x = np.zeros(n)
    residuals = []
    converged = False
    diverged = False

    for k in range(max_iter):
        res_norm = np.linalg.norm(b - A @ x)
        residuals.append(res_norm)

        if res_norm < tol:
            converged = True
            break

        if res_norm > DIVERGE_THRESHOLD or np.isnan(res_norm) or np.isinf(res_norm):
            diverged = True
            break

        for i in range(n):
            sigma = 0.0
            for j in range(n):
                if j != i:
                    sigma += A[i, j] * x[j]
            x[i] = (1 - omega) * x[i] + (omega / A[i, i]) * (b[i] - sigma)

    final_r = np.linalg.norm(b - A @ x)
    if not (np.isnan(final_r) or np.isinf(final_r)):
        residuals.append(final_r)

    status = "converged" if converged else ("diverged" if diverged else "max_iter")
    return x, residuals, status

def save_report_pdf(A, b, x_true, omega, rho_T, formula_ok,
                    x_sor, residuals, status, filename="sor_report.pdf"):
    n = A.shape[0]
    with PdfPages(filename) as pdf:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")

        L = []
        L.append("ОТЧЁТ: Метод верхней релаксации (SOR)")
        L.append("=" * 52)
        L.append("")
        L.append(f"Размерность: n = {n}")
        L.append("")

        # матрица (для больших — только фрагмент)
        if n <= 10:
            L.append("Матрица A:")
            for i in range(n):
                row = "  " + "  ".join(f"{A[i,j]:10.4f}" for j in range(n))
                L.append(row)
            L.append("")
            L.append("Вектор b:")
            L.append("  " + "  ".join(f"{bi:10.4f}" for bi in b))
        else:
            L.append(f"Матрица A: {n}x{n} (слишком большая для отчёта)")
        L.append("")
        L.append("-" * 52)

        if rho_T is not None:
            L.append(f"ρ(T = I − D⁻¹A) = {rho_T:.8f}")
        else:
            L.append("ρ(T) — не вычислен (нули на диагонали)")

        if formula_ok:
            L.append(f"Формула оптимального ω применима")
        else:
            L.append(f"Формула ω НЕ применима (ρ ≥ 1), используем ω = {omega:.4f}")

        L.append(f"ω = {omega:.8f}")
        L.append("")

        if status == "converged":
            L.append(f"Статус: СОШЁЛСЯ за {len(residuals)-1} итераций")
        elif status == "diverged":
            L.append(f"Статус: РАСХОДИТСЯ (остановлено на итерации {len(residuals)-1})")
        else:
            L.append(f"Статус: достигнут max_iter = {len(residuals)-1}")

        L.append(f"Начальная невязка ||r₀|| = {residuals[0]:.4e}")
        L.append(f"Финальная невязка ||r||  = {residuals[-1]:.4e}")
        L.append("")

        if status == "converged":
            L.append("Найденное решение x:")
            for i in range(min(n, 20)):
                L.append(f"  x[{i}] = {x_sor[i]:18.10f}")
            if n > 20:
                L.append(f"  ... (ещё {n-20} компонент)")

            if x_true is not None:
                L.append("")
                L.append("Точное решение x_true:")
                for i in range(min(n, 20)):
                    L.append(f"  x[{i}] = {x_true[i]:18.10f}")
                L.append("")
                L.append(f"||x_sor − x_true|| = "
                         f"{np.linalg.norm(x_sor - x_true):.4e}")

            L.append("")
            L.append(f"Проверка: ||Ax − b|| = "
                      f"{np.linalg.norm(A @ x_sor - b):.4e}")
        else:
            L.append("Решение не получено — метод не сошёлся.")
            L.append("Рекомендации:")
            L.append("  • Используйте СПО или диаг. преобл. матрицу")
            L.append("  • Попробуйте другое значение ω ∈ (0, 2)")

        L.append("")
        L.append("=" * 52)

        ax.text(0.02, 0.98, "\n".join(L), transform=ax.transAxes,
                fontsize=9, fontfamily="monospace",
                verticalalignment="top")
        pdf.savefig(fig)
        plt.close(fig)

    print(f"  Отчёт → {filename}")

def save_convergence_pdf(residuals, omega, status, filename="sor_convergence.pdf"):
    """График невязки (лог. масштаб) → PDF."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(range(len(residuals)), residuals, linewidth=1.6, color="royalblue")
    ax.set_xlabel("Итерация", fontsize=12)
    ax.set_ylabel(r"$\|r\| = \|b - Ax\|$", fontsize=12)

    if status == "diverged":
        title = f"SOR: РАСХОДИМОСТЬ  (ω = {omega:.4f})"
        ax.set_title(title, fontsize=13, color="red")
    else:
        title = f"SOR: сходимость невязки  (ω = {omega:.4f})"
        ax.set_title(title, fontsize=13)

    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  График → {filename}")

def main():
    print("\n  1 — случайная СПО-матрица")
    print("  2 — произвольная случайная матрица")
    print("  3 — из файла slau.txt")
    choice = input("Выбор [1]: ").strip() or "1"

    if choice == "3":
        A, b, x_true = read_system_from_file()
    elif choice == "2":
        n = int(input("n [5]: ").strip() or "5")
        seed = int(input("seed [42]: ").strip() or "42")
        A, b, x_true = generate_random(n, seed)
    else:
        n = int(input("n [5]: ").strip() or "5")
        seed = int(input("seed [42]: ").strip() or "42")
        A, b, x_true = generate_random_spd(n, seed)

    if np.any(np.abs(np.diag(A)) < 1e-15):
        print("[ОШИБКА] На диагонали A есть нули — SOR неприменим.")
        return

    omega, rho_T, formula_ok = compute_optimal_omega(A)

    if not formula_ok:
        print(f"ρ(T) = {rho_T} ≥ 1 — формула ω неприменима, ω = {omega}")
        print(f"Сходимость НЕ гарантирована.")

    x_sor, residuals, status = sor(A, b, omega=omega)

    save_report_pdf(A, b, x_true, omega, rho_T, formula_ok,
                    x_sor, residuals, status)
    save_convergence_pdf(residuals, omega, status)

    print(f"\n  ω = {omega:.6f} | итераций: {len(residuals)-1} | "
          f"статус: {status} | ||r|| = {residuals[-1]:.2e}")
    if x_true is not None and status == "converged":
        print(f"  ||x − x_true|| = {np.linalg.norm(x_sor - x_true):.2e}")
    print()


if __name__ == "__main__":
    main()
