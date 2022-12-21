import numpy as np

def NearZero(z):
    #Determines whether a scalar is small enough to be treated as zero
    return abs(z) < 1e-6

def Normalize(V):
    #Normalizes a vector

    return V / np.linalg.norm(V)

def RInv(R):
    #Inverts a rotation matrix
    return np.array(R).T

def VToso3(omg):
    #Converts a 3-vector to an so(3) representation
    return np.array([[0,      -omg[2],  omg[1]],
                     [omg[2],       0, -omg[0]],
                     [-omg[1], omg[0],       0]])

def So3ToV(so3mat):
    #Converts an so(3) representation to a 3-vector

    return np.array([so3mat[2][1], so3mat[0][2], so3mat[1][0]])

def AxisAng3(expc3):
    #Converts a 3-vector of exponential coordinates for rotation into
    return (Normalize(expc3), np.linalg.norm(expc3))

def MatExp3(so3mat):
    #Computes the matrix exponential of a matrix in so(3)

    omgtheta = So3ToV(so3mat)
    if NearZero(np.linalg.norm(omgtheta)):
        return np.eye(3)
    else:
        theta = AxisAng3(omgtheta)[1]
        omgmat = so3mat / theta
        return np.eye(3) + np.sin(theta) * omgmat \
               + (1 - np.cos(theta)) * np.dot(omgmat, omgmat)

def MatLog3(R):
    #Computes the matrix logarithm of a rotation matrix

    acosinput = (np.trace(R) - 1) / 2.0
    if acosinput >= 1:
        return np.zeros((3, 3))
    elif acosinput <= -1:
        if not NearZero(1 + R[2][2]):
            omg = (1.0 / np.sqrt(2 * (1 + R[2][2]))) \
                  * np.array([R[0][2], R[1][2], 1 + R[2][2]])
        elif not NearZero(1 + R[1][1]):
            omg = (1.0 / np.sqrt(2 * (1 + R[1][1]))) \
                  * np.array([R[0][1], 1 + R[1][1], R[2][1]])
        else:
            omg = (1.0 / np.sqrt(2 * (1 + R[0][0]))) \
                  * np.array([1 + R[0][0], R[1][0], R[2][0]])
        return VToso3(np.pi * omg)
    else:
        theta = np.arccos(acosinput)
        return theta / 2.0 / np.sin(theta) * (R - np.array(R).T)

def RpT(R, p):
   
    return np.r_[np.c_[R, p], [[0, 0, 0, 1]]]

def TRp(T):
    #Converts a homogeneous transformation matrix into a rotation matrix
  
    T = np.array(T)
    return T[0: 3, 0: 3], T[0: 3, 3]

def TInv(T):
    #Inverts a homogeneous transformation matrix

    R, p = TRp(T)
    Rt = np.array(R).T
    return np.r_[np.c_[Rt, -np.dot(Rt, p)], [[0, 0, 0, 1]]]

def VToSe3(V):
    #Converts a spatial velocity vector into a 4x4 matrix in se3

    return np.r_[np.c_[VToso3([V[0], V[1], V[2]]), [V[3], V[4], V[5]]],
                 np.zeros((1, 4))]

def Se3ToV(se3mat):
    #Converts an se3 matrix into a spatial velocity vector

    return np.r_[[se3mat[2][1], se3mat[0][2], se3mat[1][0]],
                 [se3mat[0][3], se3mat[1][3], se3mat[2][3]]]

def Adjoint(T):
    #Computes the adjoint representation of a homogeneous transformation

    R, p = TRp(T)
    return np.r_[np.c_[R, np.zeros((3, 3))],
                 np.c_[np.dot(VToso3(p), R), R]]

def ScrewToAxis(q, s, h):
    #Takes a parametric description of a screw axis and converts it to a

    return np.r_[s, np.cross(q, s) + np.dot(h, s)]

def AxisAng6(expc6):
    #Converts a 6-vector of exponential coordinates into screw axis-angle
    theta = np.linalg.norm([expc6[0], expc6[1], expc6[2]])
    if NearZero(theta):
        theta = np.linalg.norm([expc6[3], expc6[4], expc6[5]])
    return (np.array(expc6 / theta), theta)

def MatExp6(se3mat):
    #Computes the matrix exponential of an se3 representation of

    se3mat = np.array(se3mat)
    omgtheta = So3ToV(se3mat[0: 3, 0: 3])
    if NearZero(np.linalg.norm(omgtheta)):
        return np.r_[np.c_[np.eye(3), se3mat[0: 3, 3]], [[0, 0, 0, 1]]]
    else:
        theta = AxisAng3(omgtheta)[1]
        omgmat = se3mat[0: 3, 0: 3] / theta
        return np.r_[np.c_[MatExp3(se3mat[0: 3, 0: 3]),
                           np.dot(np.eye(3) * theta \
                                  + (1 - np.cos(theta)) * omgmat \
                                  + (theta - np.sin(theta)) \
                                    * np.dot(omgmat,omgmat),
                                  se3mat[0: 3, 3]) / theta],
                     [[0, 0, 0, 1]]]

def MatLog6(T):
    #Computes the matrix logarithm of a homogeneous transformation matrix
    
    R, p = TRp(T)
    omgmat = MatLog3(R)
    if np.array_equal(omgmat, np.zeros((3, 3))):
        return np.r_[np.c_[np.zeros((3, 3)),
                           [T[0][3], T[1][3], T[2][3]]],
                     [[0, 0, 0, 0]]]
    else:
        theta = np.arccos((np.trace(R) - 1) / 2.0)
        return np.r_[np.c_[omgmat,
                           np.dot(np.eye(3) - omgmat / 2.0 \
                           + (1.0 / theta - 1.0 / np.tan(theta / 2.0) / 2) \
                              * np.dot(omgmat,omgmat) / theta,[T[0][3],
                                                               T[1][3],
                                                               T[2][3]])],
                     [[0, 0, 0, 0]]]

def ProjSO3(mat):
    #Returns a projection of mat into SO(3)

    U, s, Vh = np.linalg.svd(mat)
    R = np.dot(U, Vh)
    if np.linalg.det(R) < 0:
    
        R[:, s[2, 2]] = -R[:, s[2, 2]]
    return R

def ProjSE3(mat):
    #Returns a projection of mat into SE(3)

    mat = np.array(mat)
    return RpT(ProjSO3(mat[:3, :3]), mat[:3, 3])

def DistanceToSO3(mat):
    #Returns the Frobenius norm to describe the distance of mat from the

    if np.linalg.det(mat) > 0:
        return np.linalg.norm(np.dot(np.array(mat).T, mat) - np.eye(3))
    else:
        return 1e+9

def DistSE3(mat):
    #Returns the Frobenius norm to describe the distance of mat from the SE(3) manifold

    matR = np.array(mat)[0: 3, 0: 3]
    if np.linalg.det(matR) > 0:
        return np.linalg.norm(np.r_[np.c_[np.dot(np.transpose(matR), matR),
                                          np.zeros((3, 1))],
                              [np.array(mat)[3, :]]] - np.eye(4))
    else:
        return 1e+9

def TestIfSO3(mat):
    #Returns true if mat is close to or on the manifold SO(3)

    return abs(DistanceToSO3(mat)) < 1e-3

def TestIfSE3(mat):
    #Returns true if mat is close to or on the manifold SE(3)
   
    return abs(DistSE3(mat)) < 1e-3



def FKB(M, Blist, thetalist):
    #Computes forward kinematics in the body frame for an open chain robot
    
    T = np.array(M)
    for i in range(len(thetalist)):
        T = np.dot(T, MatExp6(VToSe3(np.array(Blist)[:, i] \
                                          * thetalist[i])))
    return T

def FKS(M, Slist, thetalist):
    #Computes forward kinematics in the space frame for an open chain robot
    
    T = np.array(M)
    for i in range(len(thetalist) - 1, -1, -1):
        T = np.dot(MatExp6(VToSe3(np.array(Slist)[:, i] \
                                       * thetalist[i])), T)
    return T


def JB(Blist, thetalist):
    #Computes the body Jacobian for an open chain robot
 
    Jb = np.array(Blist).copy().astype(np.float)
    T = np.eye(4)
    for i in range(len(thetalist) - 2, -1, -1):
        T = np.dot(T,MatExp6(VToSe3(np.array(Blist)[:, i + 1] \
                                         * -thetalist[i + 1])))
        Jb[:, i] = np.dot(Adjoint(T), np.array(Blist)[:, i])
    return Jb

def JS(Slist, thetalist):
    #Computes the space Jacobian for an open chain robot
  
    Js = np.array(Slist).copy().astype(np.float)
    T = np.eye(4)
    for i in range(1, len(thetalist)):
        T = np.dot(T, MatExp6(VToSe3(np.array(Slist)[:, i - 1] \
                                * thetalist[i - 1])))
        Js[:, i] = np.dot(Adjoint(T), np.array(Slist)[:, i])
    return Js


def IKB(Blist, M, T, thetalist0, eomg, ev):
    #Computes inverse kinematics in the body frame for an open chain robot
  
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = 20
    Vb = Se3ToV(MatLog6(np.dot(TInv(FKB(M, Blist, \
                                                      thetalist)), T)))
    err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg \
          or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev
    while err and i < maxiterations:
        thetalist = thetalist \
                    + np.dot(np.linalg.pinv(JB(Blist, \
                                                         thetalist)), Vb)
        i = i + 1
        Vb \
        = Se3ToV(MatLog6(np.dot(TInv(FKB(M, Blist, \
                                                       thetalist)), T)))
        err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg \
              or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev
    return (thetalist, not err)

def IKS(Slist, M, T, thetalist0, eomg, ev):
    #Computes inverse kinematics in the space frame for an open chain robot

    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = 20
    Tsb = FKS(M,Slist, thetalist)
    Vs = np.dot(Adjoint(Tsb), \
                Se3ToV(MatLog6(np.dot(TInv(Tsb), T))))
    err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > eomg \
          or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > ev
    while err and i < maxiterations:
        thetalist = thetalist \
                    + np.dot(np.linalg.pinv(JS(Slist, \
                                                          thetalist)), Vs)
        i = i + 1
        Tsb = FKS(M, Slist, thetalist)
        Vs = np.dot(Adjoint(Tsb), \
                    Se3ToV(MatLog6(np.dot(TInv(Tsb), T))))
        err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > eomg \
              or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > ev
    return (thetalist, not err)



def ad(V):
    #Calculate the 6x6 matrix [adV] of the given 6-vector

    omgmat = VToso3([V[0], V[1], V[2]])
    return np.r_[np.c_[omgmat, np.zeros((3, 3))],
                 np.c_[VToso3([V[3], V[4], V[5]]), omgmat]]

def EulerStep(thetalist, dthetalist, ddthetalist, dt):
    #Compute the joint angles and velocities at the next timestep using first order Euler integration

    return thetalist + dt * np.array(dthetalist), \
           dthetalist + dt * np.array(ddthetalist)


def CubicTimeScaling(Tf, t):
    #Computes s(t) for a cubic time scaling

    return 3 * (1.0 * t / Tf) ** 2 - 2 * (1.0 * t / Tf) ** 3

def QuinticTimeScaling(Tf, t):
    #Computes s(t) for a quintic time scaling

    return 10 * (1.0 * t / Tf) ** 3 - 15 * (1.0 * t / Tf) ** 4 \
           + 6 * (1.0 * t / Tf) ** 5

def JointTrajectory(thetastart, thetaend, Tf, N, method):
    """
    Computes a straight-line trajectory in joint space as an N×N matrix, 
    where each of the Nrows is a nn-vector of the joint variables at an instant in time.
    The first row is θstart and the Nth row is θend.  
    elapsed time = Tf/(N−1).   
    The  parametermethod equals = 3  for cubic time scaling or 5 for quintic time scaling.
    """

    N = int(N)
    timegap = Tf / (N - 1.0)
    traj = np.zeros((len(thetastart), N))
    for i in range(N):
        if method == 3:
            s = CubicTimeScaling(Tf, timegap * i)
        else:
            s = QuinticTimeScaling(Tf, timegap * i)
        traj[:, i] = s * np.array(thetaend) + (1 - s) * np.array(thetastart)
    traj = np.array(traj).T
    return traj

def ScrewTrajectory(Xstart, Xend, Tf, N, method):
    """
    Computes a trajectory as a list of N SE(3) matrices, 
    where each matrix represents the configuration of the end-effector at an instant in time.
    The first matrix is Xstart, the Nth matrix isXend, and the motion is along a constant screw axis.
    The elapsed time = Tf/(N−1).  
    The parametermethod: either 3 for a cubic time scaling or 5 for a quintic time scaling.
    """
  
    N = int(N)
    timegap = Tf / (N - 1.0)
    traj = [[None]] * N
    for i in range(N):
        if method == 3:
            s = CubicTimeScaling(Tf, timegap * i)
        else:
            s = QuinticTimeScaling(Tf, timegap * i)
        traj[i] \
        = np.dot(Xstart, MatExp6(MatLog6(np.dot(TInv(Xstart), \
                                                      Xend)) * s))
    return traj

def CartesianTrajectory(Xstart, Xend, Tf, N, method):
    """
    Computes a trajectory as a list ofN SE(3) matrices, 
    where each matrix repre-sents the configuration of the end-effector at an instant in time.  
    The first matrixis Xstart, theNth matrix is Xend, and the origin of the end-effector frame follows  a  straight  line,  
    decoupled  from  the  rotation.   
    The  elapsed  time  between each matrix is Tf/(N−1).  
    The parametermethod equals either 3 for a cubictime scaling or 5 for a quintic time scaling."""
   
    N = int(N)
    timegap = Tf / (N - 1.0)
    traj = [[None]] * N
    Rstart, pstart = TRp(Xstart)
    Rend, pend = TRp(Xend)
    for i in range(N):
        if method == 3:
            s = CubicTimeScaling(Tf, timegap * i)
        else:
            s = QuinticTimeScaling(Tf, timegap * i)
        traj[i] \
        = np.r_[np.c_[np.dot(Rstart, \
        MatExp3(MatLog3(np.dot(np.array(Rstart).T,Rend)) * s)), \
                   s * np.array(pend) + (1 - s) * np.array(pstart)], \
                   [[0, 0, 0, 1]]]
    return traj

