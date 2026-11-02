import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

functions = [
    {
        "name": "sin(x^2)",
        "f": lambda x: np.sin(x**2),
        "df": lambda x: 2 * x * np.cos(x**2)
    },
    {
        "name": "cos(sin(x))",
        "f": lambda x: np.cos(np.sin(x)),
        "df": lambda x: -np.sin(np.sin(x)) * np.cos(x)
    },
    {
        "name": "exp(sin(cos(x)))",
        "f": lambda x: np.exp(np.sin(np.cos(x))),
        "df": lambda x: np.exp(np.sin(np.cos(x))) * np.cos(np.cos(x)) * (-np.sin(x))
    },
    {
        "name": "ln(x+3)",
        "f": lambda x: np.log(x + 3),
        "df": lambda x: 1 / (x + 3)
    },
    {
        "name": "(x+3)^0.5",
        "f": lambda x: (x + 3)**0.5,
        "df": lambda x: 0.5 * (x + 3)**(-0.5)
    }
]

def method_1(f, x, h):
    return (f(x + h) - f(x)) / h

def method_2(f, x, h):
    return (f(x) - f(x - h)) / h

def method_3(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

def method_4(f, x, h):
    term1 = (4 / 3) * (f(x + h) - f(x - h)) / (2 * h)
    term2 = (1 / 3) * (f(x + 2 * h) - f(x - 2 * h)) / (4 * h)
    return term1 - term2

def method_5(f, x, h):
    term1 = (3 / 2) * (f(x + h) - f(x - h)) / (2 * h)
    term2 = (3 / 5) * (f(x + 2 * h) - f(x - 2 * h)) / (4 * h)
    term3 = (1 / 10) * (f(x + 3 * h) - f(x - 3 * h)) / (6 * h)
    return term1 - term2 + term3

methods = [
    {"name": "Method 1 (Forward)", "func": method_1},
    {"name": "Method 2 (Backward)", "func": method_2},
    {"name": "Method 3 (Central O(h^2))", "func": method_3},
    {"name": "Method 4 (O(h^4))", "func": method_4},
    {"name": "Method 5 (O(h^6))", "func": method_5}
]

x_point = 1.0
n_values = np.arange(1, 22)
h_values = 2.0 / (2.0**n_values)

for idx, target in enumerate(functions):
    f = target["f"]
    df_exact = target["df"](x_point)
    
    plt.figure(idx + 1, figsize=(10, 6))
    
    for m in methods:
        errors = []
        for h in h_values:
            df_numeric = m["func"](f, x_point, h)
            abs_err = np.abs(df_numeric - df_exact)
            errors.append(abs_err)
        
        plt.loglog(h_values, errors, '-o', label=m["name"], markersize=4)
        plt.gca().invert_xaxis()
    
    plt.title(f"Absolute Error for f(x) = {target['name']} at x = {x_point}")
    plt.xlabel("Step size (h)")
    plt.ylabel("Absolute Error")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()

for idx in range(1, 6):
    plt.figure(idx)
    plt.savefig(f'error_plot_{idx}.png', dpi=300, bbox_inches='tight')
    plt.close()
