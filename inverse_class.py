import scipy as sc
import sympy as sp
import numpy as np


class inverse_kin():
    def __init__(self,axis,trans,constraints=None)->None:
        # x=0,y=1,z=2 axis
        # constrains in rad
        # trans and axis= same lengh
        self.dof = sp.symbols(f'x0:{len(axis)}')
        self.dummy=sp.symbols('dummy')
        self.constraints=constraints
        self.rot=[
        sp.Matrix(
            [
                [1.0,        0.0,       0.0,         0.0],
                [0.0,       sp.cos(self.dummy),  -sp.sin(self.dummy),  0.0],
                [0.0,       sp.sin(self.dummy),  sp.cos(self.dummy),   0.0],
                [0.0,       0.0,        0.0,         1.0],
            ]
        ),
        sp.Matrix(
            [
                [sp.cos(self.dummy),  0.0,   sp.sin(self.dummy),  0.0],
                [0.0,        1.0,   0.0,        0.0],
                [-sp.sin(self.dummy), 0.0,   sp.cos(self.dummy),  0.0],
                [0.0,        0.0,    0.0,       1.0],
            ]
        ),
        sp.Matrix(
            [
                [sp.cos(self.dummy), -sp.sin(self.dummy), 0.0,    0.0],
                [sp.sin(self.dummy), sp.cos(self.dummy),  0.0,    0.0],
                [0.0,       0.0,        1.0,    0.0],
                [0.0,       0.0,        0.0,    1.0],
            ]
        ),
        ]
        self.trans=sp.Matrix(
            [
                [1.0,       0.0,    0.0,    self.dummy],
                [0.0,       1.0,    0.0,    0.0],
                [0.0,       0.0,    1.0,    0.0],
                [0.0,       0.0,    0.0,    1.0],
            ]
        )
        self.end_effector=self.generate_chain(axis,trans)
    def generate_chain(self,axis,trans):
        chain=sp.eye(4)
        for i in range(len(axis)):
            chain=chain@(self.rot[axis[i]].subs(self.dummy,self.dof[i]))@(self.trans.subs(self.dummy,trans[i]))
        end_effect=sp.Matrix([[0.0],[0.0],[0.0],[1.0]])

        final=chain@end_effect

        return final[:3, 0] 

    def calculate_ik(self,target_xyz,q0=None):
        n = len(self.dof)
        if q0 is None:
            q0 = np.zeros(n)

        fwd = sp.lambdify(self.dof, self.end_effector, 'numpy')
        def residual(q):
            return np.array(fwd(*q), dtype=float).ravel() - np.asarray(target_xyz, float)
        if self.constraints is not None:
            lo = [lo for lo, hi in self.constraints]
            hi = [hi for lo, hi in self.constraints]
            res = sc.optimize.least_squares(residual, q0, bounds=(lo, hi))
        else:
            res = sc.optimize.least_squares(residual, q0)
        return res.x




def main():
    kin=inverse_kin([2,1],[0,2])
    sp.pprint(kin.end_effector)
    print(kin.calculate_ik([2,0,0]))
if __name__=='__main__':
    main()


