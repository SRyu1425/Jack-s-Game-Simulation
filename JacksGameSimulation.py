import sympy as sym
def custom_latex_printer(exp,**options):
    from google.colab.output._publish import javascript
    url = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.1.1/latest.js?config=TeX-AMS_HTML"
    javascript(url=url)
    return sym.printing.latex(exp,**options)
sym.init_printing(use_latex="mathjax",latex_printer=custom_latex_printer)

import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

# Helper Functions

# so(3) "hat" and "unhat" (3x3 skew matrix <-> 3‐vector)
def so3_hat(omega):
    # Converts a 3-vector omega = [wx, wy, wz] into a 3x3 skew‐symmetric matrix
    wx, wy, wz = omega[0], omega[1], omega[2]
    return sym.Matrix([
        [0.0, -wz, wy],
        [wz, 0.0, -wx],
        [-wy, wx, 0.0]
    ])

def so3_unhat(Omega):
    # Converts a a 3x3 into 3-vector
    wx = Omega[2,1]
    wy = Omega[0,2]
    wz = Omega[1,0]
    return sym.Matrix([wx, wy, wz])

def se3_hat(xi):
    # Convert a 6-vector xi = [v_x, v_y, v_z, w_x, w_y, w_z] into a 4×4 se(3) matrix

    # Extract translational part v and rotational part w
    v = xi[0:3]
    w = xi[3:6]
    Omega = so3_hat(w)

    # Build the top 3×4 block by horizontally stacking Omega and v
    top = Omega.row_join(v)

    # Build the bottom row [0, 0, 0, 0]
    bottom = sym.Matrix([[0, 0, 0, 0]])

    # Stack them into a 4×4
    return top.col_join(bottom)

def se3_unhat(Xi):
    # 4x4 to 6 vector
    v = Xi[0:3, 3]
    omg = Xi[0:3, 0:3]
    return v.col_join(so3_unhat(omg))

def SE3(phi, tx, ty):
    # returns SE(3) matrices given a rotation angle and 2D translation vector
    # rot about z axis (in/out page)
    R = sym.Matrix([[sym.cos(phi), -sym.sin(phi), 0],
                   [sym.sin(phi),  sym.cos(phi), 0],
                   [0,  0, 1]])
    p = sym.Matrix([tx, ty, 0])
    return R.row_join(p).col_join(sym.Matrix([[0,0,0,1]]))

def SE3_numpy(phi, tx, ty):
    return np.array([
        [  np.cos(phi), -np.sin(phi),0, tx ],
        [  np.sin(phi),  np.cos(phi), 0,ty ],
        [  0,  0, 1,  0 ],
        [  0,  0, 0,  1 ]
    ])

# config vars and derivatives
t, g, len_box, len_jack, m_box, m_jack, k = sym.symbols('t, g, len_box, len_jack,m_box, m_jack, k')

x_box = sym.Function(r'x_box')(t)
y_box = sym.Function(r'y_box')(t)
x_jack = sym.Function(r'x_jack')(t)
y_jack = sym.Function(r'y_jack')(t)
th_box = sym.Function(r'\theta_box')(t)
th_jack = sym.Function(r'\theta_jack')(t)

x_box_dot = x_box.diff(t)
y_box_dot = y_box.diff(t)
x_jack_dot = x_jack.diff(t)
y_jack_dot = y_jack.diff(t)
th_box_dot = th_box.diff(t)
th_jack_dot = th_jack.diff(t)

q = sym.Matrix([x_box, y_box, th_box, x_jack, y_jack, th_jack])
qdot = q.diff(t)
qddot = qdot.diff(t)

# Constants
m_box = 100
m_jack = 1
len_jack = 1 # full width
len_box = 10 # full width
k = 100
g= 9.81

# print('\n\033[1mRigid Body Transformation from World to Box Center Frame: ')
# display(g_W_b5)

# print('\n\033[1mWorld to Top Box Wall Frame: ')
# display(g_b5_b1)

# print('\n\033[1mWorld to Right Box Wall Frame: ')
# display(g_b5_b2)

# print('\n\033[1mWorld to Bottom Box Wall Frame: ')
# display(g_b5_b3)

# print('\n\033[1mWorld to Left Box Wall Frame: ')
# display(g_b5_b4)

# print('\n\033[1mRigid Body Transformation from World to Jack Center Frame: ')
# display(g_W_j5)

# print('\n\033[1mWorld to Jack Top Frame: ')
# display(g_j5_j1)

# print('\n\033[1mWorld to Jack Right Frame: ')
# display(g_j5_j2)

# print('\n\033[1mWorld to Jack Bottom Frame: ')
# display(g_j5_j3)

# print('\n\033[1mWorld to Jack Left Frame: ')
# display(g_j5_j4)

# transformations & frames

# world to origins
g_W_b5 = SE3(th_box, x_box, y_box)
g_W_j5 = SE3(th_jack, x_jack, y_jack)


# box frames
g_b5_b2 = SE3(0, len_box/2, 0)
g_b5_b4 = SE3(0, -len_box/2, 0)
g_b5_b1 = SE3(0, 0, len_box/2)
g_b5_b3 = SE3(0, 0, -len_box/2)

# jack frames
g_j5_j2 = SE3(0, len_jack/2, 0)
g_j5_j4 = SE3(0, -len_jack/2, 0)
g_j5_j1 = SE3(0, 0, len_jack/2)
g_j5_j3 = SE3(0, 0, -len_jack/2)


# body velocities
V_W_b5 = se3_unhat(g_W_b5.inv() * g_W_b5.diff(t))
V_W_j5 = se3_unhat(g_W_j5.inv() * g_W_j5.diff(t))

# rotational inertia
# Summation (mr**2) for point mass (jack)
# square inertia for box
J_box = m_box * (len_box/2)**2
J_jack = 4 * ((m_jack/4) * (len_jack/2)**2) # 4 point masses at each leg

# inertia tensor (planar rotations simplifies it)
I_b = sym.Matrix([[m_box, 0, 0, 0, 0, 0],
                  [0, m_box, 0, 0, 0, 0],
                  [0, 0, m_box, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, J_box]])
I_j = sym.Matrix([[m_jack, 0, 0, 0, 0, 0],
                  [0, m_jack, 0, 0, 0, 0],
                  [0, 0, m_jack, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, J_jack]])

# compute KE, PE, Lagrangian
KE_box = (1/2 * (V_W_b5.T) * I_b * V_W_b5)[0] #[0] extract scalar of 1x1 matrix
KE_jack = (1/2 * (V_W_j5.T) * I_j * V_W_j5)[0] #[0] extract scalar of 1x1 matrix
V  = m_box*g*y_box + m_jack*g*y_jack
Lagr = (KE_box + KE_jack) - V
Lagr = sym.simplify(sym.expand(Lagr))
# display(Lagr)

# External forces and EL Eqs
th_d_box = 100 *sym.sin(2*t) # rotate box
F_box_y = m_box*g# gravitational force mg (to make it stationary while jack falls)
F_box_x = 1000 * sym.cos(2*t) # oscillate side to side
F_theta_box = k*th_d_box # torque on box


Fext = sym.Matrix([F_box_x, F_box_y, F_theta_box, 0, 0, 0]) # no ext forces on jack

lhs = Lagr.diff(qdot).diff(t) - Lagr.diff(q)
rhs = Fext
EL = sym.Eq(lhs, rhs)
soln = sym.solve(EL, qddot, dict=True)

x_box_dd = soln[0][qddot[0]]
y_box_dd = soln[0][qddot[1]]
th_box_dd = soln[0][qddot[2]]

x_jack_dd = soln[0][qddot[3]]
y_jack_dd = soln[0][qddot[4]]
th_jack_dd = soln[0][qddot[5]]
display(soln[0])

# Note that the jack is dictated only by gravity
# The box has an external force pushing it side to side (sinusoidally)
# As well as a torque that rotates it (sinusoidally)

# Lambdifying above solns

func_x_box_ddot = sym.lambdify((q[0], q[1], q[2], q[3], q[4],q[5],
                                qdot[0], qdot[1], qdot[2], qdot[3], qdot[4], qdot[5], t),x_box_dd)
func_y_box_ddot = sym.lambdify((q[0], q[1], q[2], q[3], q[4],q[5],
                                qdot[0], qdot[1], qdot[2], qdot[3], qdot[4], qdot[5], t),y_box_dd)
func_th_box_ddot = sym.lambdify((q[0], q[1], q[2], q[3], q[4],q[5],
                                 qdot[0], qdot[1], qdot[2], qdot[3], qdot[4], qdot[5], t),th_box_dd)

func_x_jack_ddot = sym.lambdify((q[0], q[1], q[2], q[3], q[4],q[5],
                                 qdot[0], qdot[1], qdot[2], qdot[3], qdot[4], qdot[5], t),x_jack_dd)
func_y_jack_ddot = sym.lambdify((q[0], q[1], q[2], q[3], q[4],q[5],
                                 qdot[0], qdot[1], qdot[2], qdot[3], qdot[4], qdot[5], t),y_jack_dd)
func_th_jack_ddot = sym.lambdify((q[0], q[1], q[2], q[3], q[4],q[5],
                                  qdot[0], qdot[1], qdot[2], qdot[3], qdot[4], qdot[5], t),th_jack_dd)

def sdot(s, t):
    """
    12-dimensional state vector s = [ xb, yb, thb, xj, yj, thj, xdb, ydb, thdb, xdj, ydj, thdj]
    and time t
    Return sdot
    """
    return np.array([
        s[6],
        s[7],
        s[8],
        s[9],
        s[10],
        s[11],
        func_x_box_ddot(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9], s[10], s[11], t),
        func_y_box_ddot(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9], s[10], s[11], t),
        func_th_box_ddot(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9], s[10], s[11], t),
        func_x_jack_ddot(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9], s[10], s[11], t),
        func_y_jack_ddot(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9], s[10], s[11], t),
        func_th_jack_ddot(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9], s[10], s[11], t),
    ])

"""$$g_{b1j1} = g_{b5b1}^{-1} g_{Wb5}^{-1} g_{Wj5} g_{j5j1}$$"""

# Constraints phi for eventual impact equations

# Each wall will have 4 different constraint eqs
# Each end of the jack must not pass through each wall
# 4 jack ends * 4 walls = 16 eqs.

# wall 1 (top wall) & each jack end
# location of jack end 1 in frame of wall 1
# first one: b1->b5->W->j5->j1
g_b1_j1 = g_b5_b1.inv() * g_W_b5.inv() * g_W_j5 * g_j5_j1
g_b1_j2 = g_b5_b1.inv() * g_W_b5.inv() * g_W_j5 * g_j5_j2
g_b1_j3 = g_b5_b1.inv() * g_W_b5.inv() * g_W_j5 * g_j5_j3
g_b1_j4 = g_b5_b1.inv() * g_W_b5.inv() * g_W_j5 * g_j5_j4

# wall 2
g_b2_j1 = g_b5_b2.inv() * g_W_b5.inv() * g_W_j5 * g_j5_j1
g_b2_j2 = g_b5_b2.inv() * g_W_b5.inv() * g_W_j5 * g_j5_j2
g_b2_j3 = g_b5_b2.inv() * g_W_b5.inv() * g_W_j5 * g_j5_j3
g_b2_j4 = g_b5_b2.inv() * g_W_b5.inv() * g_W_j5 * g_j5_j4

# wall 3
g_b3_j1 = g_b5_b3.inv() * g_W_b5.inv() * g_W_j5 * g_j5_j1
g_b3_j2 = g_b5_b3.inv() * g_W_b5.inv() * g_W_j5 * g_j5_j2
g_b3_j3 = g_b5_b3.inv() * g_W_b5.inv() * g_W_j5 * g_j5_j3
g_b3_j4 = g_b5_b3.inv() * g_W_b5.inv() * g_W_j5 * g_j5_j4

# wall 4
g_b4_j1 = g_b5_b4.inv() * g_W_b5.inv() * g_W_j5 * g_j5_j1
g_b4_j2 = g_b5_b4.inv() * g_W_b5.inv() * g_W_j5 * g_j5_j2
g_b4_j3 = g_b5_b4.inv() * g_W_b5.inv() * g_W_j5 * g_j5_j3
g_b4_j4 = g_b5_b4.inv() * g_W_b5.inv() * g_W_j5 * g_j5_j4


# Wall 1
# y coordinate of frame j (tip of jack) should not go above wall ( >0 )
phi_b1_j1= g_b1_j1[1, 3]
phi_b1_j2= g_b1_j2[1, 3]
phi_b1_j3= g_b1_j3[1, 3]
phi_b1_j4= g_b1_j4[1, 3]

# Wall 2
# x coordinate of frame j (tip of jack) should not go past (sideways right) thru wall
phi_b2_j1= g_b2_j1[0, 3]
phi_b2_j2= g_b2_j2[0, 3]
phi_b2_j3= g_b2_j3[0, 3]
phi_b2_j4= g_b2_j4[0, 3]

# Wall 3
# y coordinate of frame j (tip of jack) should not go below wall
phi_b3_j1= g_b3_j1[1, 3]
phi_b3_j2= g_b3_j2[1, 3]
phi_b3_j3= g_b3_j3[1, 3]
phi_b3_j4= g_b3_j4[1, 3]

# Wall 4
# x coordinate of frame j (tip of jack) should not go past (sideways left) thru wall
phi_b4_j1= g_b4_j1[0, 3]
phi_b4_j2= g_b4_j2[0, 3]
phi_b4_j3= g_b4_j3[0, 3]
phi_b4_j4= g_b4_j4[0, 3]

# all phi constraints
phi_matrix = sym.Matrix([phi_b1_j1, phi_b1_j2, phi_b1_j3, phi_b1_j4,
                        phi_b2_j1, phi_b2_j2, phi_b2_j3, phi_b2_j4,
                        -phi_b3_j1, -phi_b3_j2, -phi_b3_j3, -phi_b3_j4,
                        -phi_b4_j1, -phi_b4_j2, -phi_b4_j3, -phi_b4_j4])

# display(phi_matrix)

lam = sym.symbols('lambda')

# dummy vars
x_b, y_b, th_b, x_j, y_j, th_j, x_b_dot, y_b_dot, th_b_dot, x_j_dot, y_j_dot, th_j_dot = sym.symbols(r'x_b, y_b, th_b, x_j, y_j, th_j, x_b_dot, y_b_dot, th_b_dot, x_j_dot, y_j_dot, th_j_dot')
dummy_dict = {q[0]:x_b, q[1]:y_b, q[2]:th_b, q[3]:x_j, q[4]:y_j, q[5]:th_j,
              qdot[0]:x_b_dot, qdot[1]:y_b_dot, qdot[2]:th_b_dot, qdot[3]:x_j_dot, qdot[4]:y_j_dot, qdot[5]:th_j_dot}

# Get expressions used in impact Equations

Lagr_sym= Lagr.subs(dummy_dict)
# print('\n\033[1mLagrangian:')
# display(Lagr_sym)

dL_dqdot = Lagr.diff(qdot)
dL_dqdot_minus = sym.simplify(dL_dqdot.subs(dummy_dict))

# print('\n\033[1mdL/dqdot:')
# display(dL_dqdot_minus)

# constraint phi = x3 = 0
# phi = phi_matrix.subs(dummy_dict)
# display(phi.shape)
dphi_dq = phi_matrix.jacobian(q)
dphi_dq = sym.simplify(dphi_dq.subs(dummy_dict))
# display(dphi_dq.shape)
# display(dL_dqdot_minus.shape)
# display(dphi_dq)


# print('\n\033[1mdPhi/dq:')
# display(dphi_dq)

# display(dL_dqdot_minus.shape) #3,1 so must transpose
# display(qdot.shape) #3,1

# hamiltonian
# print('\n\033[1mHamiltonian:')
qdot_sym = qdot.subs(dummy_dict)
M = (dL_dqdot_minus.T * qdot_sym)
p_qdot = M[0,0]
hminus = sym.simplify(p_qdot - Lagr_sym)
# display(hminus)

# Same process for tau+ (moment after impact)
# Generate impact equations

# Dummy symbols for qdot
x_b_dot_plus, y_b_dot_plus, th_b_dot_plus, x_j_dot_plus, y_j_dot_plus, th_j_dot_plus = sym.symbols('x_b_dot_plus, y_b_dot_plus, th_b_dot_plus, x_j_dot_plus, y_j_dot_plus, th_j_dot_plus')
dummy_dict_plus = {x_b_dot:x_b_dot_plus, y_b_dot:y_b_dot_plus, th_b_dot:th_b_dot_plus, x_j_dot:x_j_dot_plus, y_j_dot:y_j_dot_plus, th_j_dot:th_j_dot_plus}
# impact update eqns (eval at tau- and tau+)
# angles before and after impact are same

dL_dqdot_plus = dL_dqdot_minus.subs(dummy_dict_plus)
# display(dphi_dq.shape)
# display(dL_dqdot_plus[0,0])
# display(dL_dqdot_minus[1,0])

# display(dL_dqdot_plus.rows)
h_plus = hminus.subs(dummy_dict_plus)

impact_eqs_list = []

for phi_case in range(dphi_dq.rows):  # 0..15

    # same 6 eqs for the 6 config vars
    # 16 times for 16 phi eqs
    lhs = sym.Matrix([
        dL_dqdot_plus[0, 0] - dL_dqdot_minus[0, 0], # dL_dq = lam gradphi eqs
        dL_dqdot_plus[1, 0] - dL_dqdot_minus[1, 0],
        dL_dqdot_plus[2, 0] - dL_dqdot_minus[2, 0],
        dL_dqdot_plus[3, 0] - dL_dqdot_minus[3, 0],
        dL_dqdot_plus[4, 0] - dL_dqdot_minus[4, 0],
        dL_dqdot_plus[5, 0] - dL_dqdot_minus[5, 0],
        h_plus - hminus # H+ - H- = 0 equation
    ])

    # dphi_dq (16,6) so loop over each phi case and assign to config var
    rhs = sym.Matrix([
        lam * dphi_dq[phi_case, 0],
        lam * dphi_dq[phi_case, 1],
        lam * dphi_dq[phi_case, 2],
        lam * dphi_dq[phi_case, 3],
        lam * dphi_dq[phi_case, 4],
        lam * dphi_dq[phi_case, 5],
        0
    ])

    impact_eqs_list.append(sym.simplify(sym.Eq(lhs, rhs)))
# display(len(impact_eqs_list))

# print('\n\033[1mImpact Equations:')
# for eq in impact_eqs:
#     display(eq)

# functions for use in impact simulation


def impact_update(s_old, impact_eqs): #updates state variable after impact
    num_subs_dict = {x_b : s_old[0], y_b:s_old[1], th_b:s_old[2], x_j: s_old[3], y_j:s_old[4], th_j:s_old[5],
                     x_b_dot:s_old[6], y_b_dot: s_old[7], th_b_dot:s_old[8], x_j_dot:s_old[9], y_j_dot:s_old[10], th_j_dot:s_old[11]}

    new_impact_eqs = impact_eqs.subs(num_subs_dict).evalf()

    solns = sym.solve(new_impact_eqs, [x_b_dot_plus, y_b_dot_plus, th_b_dot_plus,
                                       x_j_dot_plus, y_j_dot_plus, th_j_dot_plus,
                                       lam], dict=True)

    # print('\n\033[1mSolution:')


    # display(solns)

    for sol in solns:
      lambd = sol[lam]
      # print(lambd)
    # display(solns[0]) #lambda = 0 so impact not taken into account
    # display(solns[1]) #this seems to be the correct soln for lambda
    # display(sym.Eq(x_b_dot_plus, solns[1][x_b_dot_plus]))
    # display(sym.Eq(lam, solns[1][lam]))
      if abs(lambd) > 0:
        # print(sol)

        return np.array([s_old[0],s_old[1],s_old[2],s_old[3], s_old[4], s_old[5],
                      float(sym.N(sol[x_b_dot_plus])), float(sym.N(sol[y_b_dot_plus])), float(sym.N(sol[th_b_dot_plus])),
                      float(sym.N(sol[x_j_dot_plus])), float(sym.N(sol[y_j_dot_plus])), float(sym.N(sol[th_j_dot_plus]))])
    print("err")
    return np.array([0,0,0,0,0,0,0,0,0,0,0,0])




#test result
# a, b, c, d, e, f, g, h, i, j, k, l = impact_update([0,0,0,0,0,0,-10,-10,-100,-10,-10,-10], impact_eqs_list[0])



func_phi = sym.lambdify([[q[0],q[1],q[2], q[3], q[4], q[5],qdot[0],qdot[1],qdot[2], qdot[3],qdot[4],qdot[5]]], phi_matrix)
# threshold = .1
def impact_condition(s):
    """
    if phi is within threshold, impact
    return true if impact
    and which of the 16 phi eqs is the impact
    """

    # Evaluate all 16 phi_i at the curr state
    phi = func_phi([
        s[0], s[1], s[2],
        s[3], s[4], s[5],
        s[6], s[7], s[8],
        s[9], s[10], s[11]
    ])

    # Both phi_before and phi_after are length‑16 arrays
    # Now loop over each index i = 0..15 and check within threshold
    for i in range(len(phi)):  # len = 16
        if (phi[i] > 0):
            return (True, i)

    # If no index triggered, return False
    return (False, None)


def integrate(f, xt, dt, t):
    """
    This function takes in an initial condition x(t) and a timestep dt,
    as well as a dynamical system f(x) that outputs a vector of the
    same dimension as x(t). It outputs a vector x(t+dt) at the future
    time step.

    Parameters
    ============
    dyn: Python function
        derivate of the system at a given step x(t),
        it can considered as \dot{x}(t) = func(x(t))
    xt: NumPy array
        current step x(t)
    dt:
        step size for integration
    t: current ime

    Return
    ============
    new_xt:
        value of x(t+dt) integrated from x(t)
    """
    k1 = dt * f(xt, t)
    k2 = dt * f(xt+k1/2., t)
    k3 = dt * f(xt+k2/2., t)
    k4 = dt * f(xt+k3, t)
    new_xt = xt + (1/6.) * (k1+2.0*k2+2.0*k3+k4)
    return new_xt


def impact_simulate(f, x0, tspan, dt, integrate):
    """
    This function takes in an initial condition x0, a timestep dt,
    a time span tspan consisting of a list [min_time, max_time],
    as well as a dynamical system f(x) that outputs a vector of the
    same dimension as x0. It outputs a full trajectory simulated
    over the time span of dimensions (xvec_size, time_vec_size).

    Parameters
    ============
    f: Python function
        derivate of the system at a given step x(t),
        it can considered as \dot{x}(t) = func(x(t))
    x0: NumPy array
        initial conditions (th, thdot)
    tspan: Python list
        tspan = [min_time, max_time], it defines the start and end
        time of simulation
    dt:
        time step for numerical integration
    integrate: Python function
        numerical integration method used in this simulation

    Return
    ============
    x_traj:
        simulated trajectory of x(t) from t=0 to tf
    """
    N = int((max(tspan)-min(tspan))/dt)
    x = np.copy(x0)
    tvec = np.linspace(min(tspan),max(tspan),N)
    xtraj = np.zeros((len(x0),N+1))
    xtraj[:,0] = x #initialize the trajectory with initial conditions

    for i in range(N):
        # get the next state by integrating along curr traj
        x_next = integrate(f,x,dt, tvec[i])
        test, row = impact_condition(x_next)

        if test: #check if theres a sign change
            # sign change = impact
            x_next = impact_update(x, impact_eqs_list[row]) # update state variables for impact


        xtraj[:,i+1] = x_next #update trajectory
        x = np.copy(x_next) #update prev state
    return xtraj

# Simulation and trajectory graphs
N = int((10-0)/0.01)         # =1000
t_list = np.linspace(0, 10, N+1)   # length =1001: [0.00, 0.01, …, 9.99, 10.00]
s0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
traj = impact_simulate(sdot, s0, [0, 10], 0.01, integrate)
display(traj.shape)
# Plotting x and y vs. time
plt.plot(t_list, traj[(0, 1, 2), :].T)
plt.title("x,y, theta of Box vs. Time")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.legend(["x_box", "y_box", "theta_box"])
plt.show()

# Plotting theta1 and theta2 vs. time
plt.plot(t_list, traj[(3, 4, 5), :].T)
plt.title("x,y, theta of Jack vs. Time")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.legend(["x_jack", "y_jack", "theta_jack"])
plt.show()

def animate_final(q_array, Lb=5, Lj=0.5, T=10):
    #Lb half length of box
    #Lj half length of jack

    ################################
    # Imports required for animation.
    from plotly.offline import init_notebook_mode, iplot
    from IPython.display import display, HTML
    import plotly.graph_objects as go

    #######################
    # Browser configuration (only needed in certain Jupyter environments).
    def configure_plotly_browser_state():
        import IPython
        display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-1.5.1.min.js?noext',
            },
          });
        </script>
        '''))

    configure_plotly_browser_state()
    init_notebook_mode(connected=False)

    ###############################################
    # Extract the 4 corners for both box and jack

    x_box_over_time = q_array[0]
    y_box_over_time = q_array[1]
    theta_box_over_time = q_array[2]
    x_jack_over_time= q_array[3]
    y_jack_over_time = q_array[4]
    theta_jack_over_time = q_array[5]

    N = len(x_box_over_time)  # number of frames

    # Allocate arrays for the four box‐corner points at each time step:
    # b_center_x = np.zeros(N) Don't need center of box
    # b_center_y = np.zeros(N)
    b1_x = np.zeros(N)
    b1_y = np.zeros(N)
    b2_x = np.zeros(N)
    b2_y = np.zeros(N)
    b3_x = np.zeros(N)
    b3_y = np.zeros(N)
    b4_x = np.zeros(N)
    b4_y = np.zeros(N)

    # Allocate arrays for the four jack‐end points plus the center:
    j_center_x = np.zeros(N)
    j_center_y = np.zeros(N)
    j1_x = np.zeros(N)
    j1_y = np.zeros(N)
    j2_x = np.zeros(N)
    j2_y = np.zeros(N)
    j3_x = np.zeros(N)
    j3_y = np.zeros(N)
    j4_x = np.zeros(N)
    j4_y = np.zeros(N)


    # Populate all corner‐coordinates at each time step t = 0…N-1
    for t in range(N):
        # Box frame in world:
        g_WB = SE3_numpy(theta_box_over_time[t],x_box_over_time[t], y_box_over_time[t])
        # Jack frame in world:
        g_WJ = SE3_numpy(theta_jack_over_time[t],x_jack_over_time[t], y_jack_over_time[t])

        # Box’s four corners (in its own frame)
        # append a 1 so we can do g_WB.dot(..., 1]) in numpy
        # Box's corners locations in world frame
        pB1 = g_WB.dot(np.array([-Lb,  Lb, 0, 1])) # top left
        pB2 = g_WB.dot(np.array([ Lb,  Lb, 0, 1])) # top right
        pB3 = g_WB.dot(np.array([ Lb, -Lb, 0, 1])) # bottom right
        pB4 = g_WB.dot(np.array([-Lb, -Lb, 0, 1])) # bottom left

        b1_x[t], b1_y[t] = pB1[0], pB1[1]
        b2_x[t], b2_y[t] = pB2[0], pB2[1]
        b3_x[t], b3_y[t] = pB3[0], pB3[1]
        b4_x[t], b4_y[t] = pB4[0], pB4[1]

        # Same for jack
        pJc = g_WJ.dot(np.array([ 0,    0, 0, 1]))
        pJ1 = g_WJ.dot(np.array([ Lj,   0, 0, 1]))
        pJ2 = g_WJ.dot(np.array([ 0,    Lj,0, 1]))
        pJ3 = g_WJ.dot(np.array([-Lj,   0, 0, 1]))
        pJ4 = g_WJ.dot(np.array([ 0,   -Lj,0, 1]))

        j_center_x[t], j_center_y[t] = pJc[0], pJc[1]
        j1_x[t], j1_y[t] = pJ1[0], pJ1[1]
        j2_x[t], j2_y[t]= pJ2[0], pJ2[1]
        j3_x[t], j3_y[t]= pJ3[0], pJ3[1]
        j4_x[t], j4_y[t]= pJ4[0], pJ4[1]

    # Number of frames
    N = len(q_array[0])

    ####################################
    # SET AXIS LIMITS FOR PLOTTING
    xm, xM = -17, 17
    ym, yM = -17, 17

    ###########################
    # Defining data dictionary.
    data = [
        dict(name='Box'),
        dict(name='Jack'),
    ]

    ################################
    # Preparing simulation layout.
    # Title and axis ranges are here.
    layout=dict(autosize=False, width=1000, height=1000,
                xaxis=dict(range=[xm, xM], autorange=False, zeroline=False,dtick=1),
                yaxis=dict(range=[ym, yM], autorange=False, zeroline=False,scaleanchor = "x",dtick=1),
                title='Jacks Game Simulation',
                hovermode='closest',
                updatemenus= [{'type': 'buttons',
                               'buttons': [{'label': 'Play','method': 'animate',
                                            'args': [None, {'frame': {'duration': T, 'redraw': False}}]},
                                           {'args': [[None], {'frame': {'duration': T, 'redraw': False}, 'mode': 'immediate',
                                            'transition': {'duration': 0}}],'label': 'Pause','method': 'animate'}
                                          ]
                              }]
               )

    ########################################
    # BUILD EACH FRAME
    frames = []
    for k in range(N):
        # Jack and box corners
        box_x = [b1_x[k],  b2_x[k],  b3_x[k],  b4_x[k],  b1_x[k]]
        box_y = [ b1_y[k],  b2_y[k],  b3_y[k],  b4_y[k],  b1_y[k]  ]

        jack_xc = [j1_x[k],  j3_x[k],  j_center_x[k],  j4_x[k],  j2_x[k]]
        jack_yc =[j1_y[k],  j3_y[k],  j_center_y[k],  j4_y[k],  j2_y[k] ]

        jack_x = [j1_x[k],  j3_x[k],  j4_x[k],  j2_x[k]]
        jack_y =[j1_y[k],  j3_y[k], j4_y[k],  j2_y[k] ]

        frame_data = [
            dict(  # Trace for  box
                x=box_x,
                y=box_y,
                mode='lines',
                line=dict(color='red', width=3),
            ),
            dict(  # Trace for Leg 2
                x=jack_xc,
                y=jack_yc,
                mode='lines',
                line=dict(color='blue', width=3),
            ),
            go.Scatter(
                x=jack_x,
                y=jack_y,
                mode="markers",
                marker=dict(color='darkblue', size=6)
            )
        ]
        frames.append(dict(data=frame_data, name=f'frame{k}'))

    #######################################
    # Putting it all together and plotting.
    figure1 = dict(data=data, layout=layout, frames=frames)
    iplot(figure1)

animate_final(traj[(0,1,2,3,4,5), :])

