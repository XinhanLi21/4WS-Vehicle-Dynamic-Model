# mpc_4ws.py  --- rear‑steer online version (2025‑05‑06)
import numpy as np
from cvxopt import matrix, solvers
import sympy as sp


# ========= vehicle + MPC parameters =======================================
class VehiclePara:
    def __init__(self):
        self.m, self.g = 1350.0, 9.8
        self.Lf, self.Lr = 1.4, 1.6
        self.Iz = 1536.0
        self.Ccf, self.Ccr = 66900.0, 62700.0


class MPCParameters:
    def __init__(self, Ts=0.05, Np=20, Nc=5):
        self.Ts, self.Np, self.Nc = Ts, Np, Nc


# ========= MPC controller ==================================================
class MPC4WS:
    """
    Discrete‑time MPC for 4‑wheel‑steer vehicle.
    * front‑wheel steer δ_f is the control variable (optimised),
    * rear‑wheel steer δ_r is a **measured disturbance**, fed in every step.

    The caller passes δ_r (deg) in u[6]; if absent, defaults to 0.
    """

    def __init__(self, path_ref, Ts: float = 0.05, vx: float = 10.0):
        self.param = VehiclePara()
        self.mpc   = MPCParameters(Ts)
        self.vx    = float(vx)          # constant longitudinal speed in model
        self.U     = 0.0                # previous δ_f
        self.xd    = np.array([1e-3, 1e-4, 1e-4, 1e-5])

        self.path = np.asarray(path_ref, float)
        if self.path.ndim != 2 or self.path.shape[1] != 3:
            raise ValueError("path_ref must be (N,3) = [X,Y,psi(rad)].")

        self._build_symbolic_jacobians()

    # ------------------------------------------------------------------
    def reset(self):
        self.U  = 0.0
        self.xd = np.array([1e-3, 1e-4, 1e-4, 1e-5])

    # ------------------------------------------------------------------
    def step(self, t, u):
        """
        u = [y_dot(km/h), x_dot(km/h), phi(deg), r(deg/s),
             Y(m), X(m), delta_r(deg) , reserve]
        delta_r 可省略 -> 默认 0
        """
        # ----- 1. measured states -----------------------------------
        y_dot = u[0] / 3.6
        x_dot = u[1] / 3.6 + 1e-4
        phi   = np.deg2rad(u[2])
        r     = np.deg2rad(u[3])
        Y_pos, X_pos = u[4], u[5]
        delta_r = np.deg2rad(u[6]) if len(u) > 6 else 0.0   # ★ rear‑steer

        # ----- 2. augmented state ----------------------------------
        kesi = np.array([phi, y_dot, Y_pos, r, self.U])

        # ----- 3. local‑linear model & disturbance -----------------
        A, b = self._model_jacobian(kesi)
        x1   = self._f_forward(phi, y_dot, Y_pos, r,
                               delta_f=self.U, delta_r=delta_r)
        d_k  = x1 - A @ kesi[:4] - b * kesi[4]
        d_aug = np.hstack([d_k, 0.0])           # ((Nx+Nu),)

        # ----- 4. batch matrices -----------------------------------
        Nx, Nu, Ny = 4, 1, 2
        Np, Nc, Ts = self.mpc.Np, self.mpc.Nc, self.mpc.Ts

        Aaug = np.block([[A,               b.reshape(-1, 1)],
                         [np.zeros((Nu, Nx)), np.eye(Nu)]])
        Baug = np.vstack([b.reshape(-1, 1), np.ones((Nu, 1))])
        Cmat = np.array([[1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0]])

        PSI, GAMMA, THETA, PHI = [], [], [], []
        for p in range(1, Np + 1):
            PHI.append(d_aug)
            rowG, rowT = [], []
            for q in range(1, Np + 1):
                G = Cmat @ np.linalg.matrix_power(Aaug, p - q) \
                    if q <= p else np.zeros((Ny, Nx + Nu))
                rowG.append(G)
            for k in range(1, Nc + 1):
                Tmat = Cmat @ np.linalg.matrix_power(Aaug, p - k) @ Baug \
                       if k <= p else np.zeros((Ny, 1))
                rowT.append(Tmat)
            GAMMA.append(np.hstack(rowG))
            THETA.append(np.hstack(rowT))
            PSI.append(Cmat @ np.linalg.matrix_power(Aaug, p))

        PSI   = np.vstack(PSI)
        GAMMA = np.vstack(GAMMA)
        THETA = np.vstack(THETA)
        PHI   = np.vstack(PHI).reshape(-1, 1)

        # ----- 5. reference ----------------------------------------
        vx_inert = x_dot*np.cos(phi) - y_dot*np.sin(phi)
        X_pred = X_pos + np.arange(1, Np+1)*Ts*vx_inert
        Y_ref  = np.interp(X_pred, self.path[:,0], self.path[:,1])
        psi_ref= np.interp(X_pred, self.path[:,0], self.path[:,2])

        Yita_ref = np.vstack([psi_ref, Y_ref]).T.reshape(-1,1)
        err1 = Yita_ref - PSI @ kesi.reshape(-1,1) - GAMMA @ PHI

        # ----- 6. quadratic program --------------------------------
        Qblk = np.kron(np.eye(Np), np.diag([3000., 5000.]))
        Rblk = 5e4*np.eye(Nu*Nc)
        H = 2*(THETA.T @ Qblk @ THETA + Rblk)
        f = (-2*err1.T @ Qblk @ THETA).flatten()

        A_t  = np.tril(np.ones((Nc, Nc)))
        A_I  = np.kron(A_t, 1)
        Ut   = self.U*np.ones((Nc,1))
        umin, umax = -0.3744, 0.3744
        dumin, dumax = -0.248, 0.248
        A_cons = np.vstack([A_I, -A_I])
        b_cons = np.vstack([umax*np.ones((Nc,1))-Ut,
                            -umin*np.ones((Nc,1))+Ut])
        lb, ub = dumin*np.ones((Nc,1)), dumax*np.ones((Nc,1))

        du_vec = self._solve_qp(H, f, A_cons, b_cons, lb, ub)
        du = du_vec[0] if du_vec is not None else 0.0

        # ----- 7. update & output ----------------------------------
        self.U += du
        lat_err = Y_pos - Y_ref[0]
        yaw_err = phi   - psi_ref[0]
        return np.array([self.U, 0.0, lat_err, yaw_err])

    # ==============================================================
    # internal helpers ---------------------------------------------
    def _build_symbolic_jacobians(self):
        """Create lambdas A(phi,y,r), B(phi,y,r) for given vx."""
        phi, y_dot, r, delta_f = sp.symbols('phi y_dot r delta_f')
        p, Ts, vx = self.param, self.mpc.Ts, self.vx

        dy_dot = (-(p.Ccf+p.Ccr)*y_dot - (p.Lf*p.Ccf-p.Lr*p.Ccr)*r)/(p.m*vx) \
                 + vx*r + p.Ccf*delta_f/p.m
        dr = (-(p.Lf*p.Ccf-p.Lr*p.Ccr)*y_dot
              - (p.Lf**2*p.Ccf + p.Lr**2*p.Ccr)*r)/(p.Iz*vx) \
             + p.Lf*p.Ccf*delta_f/p.Iz
        Y_dot = vx*phi + y_dot

        f_vec = sp.Matrix([r, dy_dot, Y_dot, dr])
        A_c = f_vec.jacobian([phi, y_dot, sp.symbols('Ypos'), r])
        B_c = f_vec.jacobian([delta_f])

        self._A = sp.lambdify((phi, y_dot, r),
                              (sp.eye(4) + Ts*A_c), 'numpy')
        self._B = sp.lambdify((phi, y_dot, r),
                              (Ts*B_c), 'numpy')

    def _model_jacobian(self, kesi):
        phi, y_dot, _, r, _ = kesi
        A = np.asarray(self._A(phi, y_dot, r), float)
        B = np.asarray(self._B(phi, y_dot, r), float).flatten()
        return A, B

    # ---------- forward Euler incl. rear‑steer --------------------
    def _f_forward(self, phi, y_dot, Y_pos, r, *, delta_f, delta_r):
        p, Ts, vx = self.param, self.mpc.Ts, self.vx
        x1 = np.zeros(4)

        x1[0] = phi + Ts * r

        x1[1] = y_dot + Ts * (-(p.Ccf+p.Ccr)*y_dot
                              - (p.Lf*p.Ccf-p.Lr*p.Ccr)*r) / (p.m*vx) \
                        + Ts * (vx*r
                                + p.Ccf*delta_f/p.m
                                + p.Ccr*delta_r/p.m)

        x1[2] = Y_pos + Ts * (vx*phi + y_dot)

        x1[3] = r + Ts * (-(p.Lf*p.Ccf-p.Lr*p.Ccr)*y_dot
                          - (p.Lf**2*p.Ccf + p.Lr**2*p.Ccr)*r) / (p.Iz*vx) \
                      + Ts * (p.Lf*p.Ccf*delta_f/p.Iz
                              - p.Lr*p.Ccr*delta_r/p.Iz)
        return x1

    # ---------- QP solver ----------------------------------------
    @staticmethod
    def _solve_qp(H, f, A, b, lb, ub):
        H = 0.5*(H + H.T)
        P, q = matrix(H), matrix(f)
        G_box = np.vstack([np.eye(len(f)), -np.eye(len(f))])
        h_box = np.vstack([ub, -lb])
        G = matrix(np.vstack([A, G_box]))
        h = matrix(np.vstack([b, h_box]))
        solvers.options['show_progress'] = False
        try:
            sol = solvers.qp(P, q, G, h)
            return np.array(sol['x']).flatten()
        except Exception:
            return None
