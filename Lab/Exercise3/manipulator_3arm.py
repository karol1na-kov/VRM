"""
## =========================================================================== ## 
MIT License
Copyright (c) 2020 Roman Parak
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
## =========================================================================== ## 
Author   : Roman Parak
Email    : Roman.Parak@outlook.com
Github   : https://github.com/rparak
File Name: manipulator.py
## =========================================================================== ## 
Additional changes: adding robot's 3rd arm (part of VRM lab)
Author            : Karolina Kovarova 
"""

# Numpy (Array computing Lib.) [pip3 install numpy]
import numpy as np
# Mtaplotlib (Visualization Lib.) [pip3 install matplotlib]
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class DH_parameters(object):
    # << DH (Denavit-Hartenberg) parameters structure >> #
    def __init__(self, theta, a, d, alpha):
        # Angle about previous z, from old x to new x
        # Unit [radian]
        self.theta = theta
        # Length of the common normal. Assuming a revolute joint, this is the radius about previous z
        # Unit [metres]
        self.a = a
        # Offset along previous z to the common normal 
        # Unit [metres]
        self.d = d
        # Angle about common normal, from old z axis to new z axis
        # Unit [radian]
        self.alpha = alpha

class Control(object):
    def __init__(self, robot_name, rDH_param, ax_wr):
        # << PUBLIC >> #
        # Robot DH (Denavit-Hartenberg) parameters 
        self.rDH_param  = rDH_param
        # Robot Name -> Does not affect functionality (only for user)
        self.robot_name = robot_name
        # Axis working range
        self.ax_wr = ax_wr
        # Translation Part -> p1(x, y) - translation of the 1st joint, p2(x, y) - translation of the 2nd joint
        self.p1 = np.zeros(2)
        self.p2 = np.zeros(2)
        # Joints Rotation -> theta(theta_1, theta_2, theta_3)
        self.theta = np.zeros(3)
        # << PRIVATE >> #
        # Transformation matrix for FK calculation [4x4]
        self.__Tn_theta   = np.matrix(np.identity(4))
        # Auxiliary variables -> Target (Translation, Joint/Rotation) for calculation FK/IK
        self.__p_target     = None
        self.__theta_target = np.zeros(3)
        # Rounding index of calculation (accuracy)
        self.__rounding_index = 10
        # Display information about the results of DH Parameters.
        self.__display_rDHp()

    def __fast_calc_fk(self):
        """
        Description: 
            Fast calculation of forward kinematics (in this case it is recommended to use).
        """

        self.p1[0] = round(self.rDH_param.a[0] * np.cos(self.rDH_param.theta[0]) + 
                    self.rDH_param.a[1] * np.cos(self.rDH_param.theta[1]), self.__rounding_index)
        self.p1[1] = round(self.rDH_param.a[0] * np.sin(self.rDH_param.theta[0]) + 
                    self.rDH_param.a[1] * np.sin(self.rDH_param.theta[1]), self.__rounding_index)

        self.p2[0] = round(self.rDH_param.a[0] * np.cos(self.rDH_param.theta[0]) + 
                    self.rDH_param.a[1] * np.cos(self.rDH_param.theta[1]) + 
                    self.rDH_param.a[2] * np.cos(self.rDH_param.theta[2]), self.__rounding_index)
        self.p2[1] = round(self.rDH_param.a[0] * np.sin(self.rDH_param.theta[0]) + 
                    self.rDH_param.a[1] * np.sin(self.rDH_param.theta[1]) +
                    self.rDH_param.a[2] * np.sin(self.rDH_param.theta[2]), self.__rounding_index)

    def forward_kinematics(self, theta, degree_repr):
        """
        Description:
            Forward kinematics refers to the use of the kinematic equations of a robot to compute 
            the position of the end-effector from specified values for the joint parameters.
            Joint Angles (Theta_1, Theta_2) <-> Position of End-Effector (x, y)
        Args:
            (1) calc_type [INT]: Select the type of calculation (0: DH Table, 1: Fast).
            (2) theta [Float Array]: Joint angle of target in degrees.
            (3) degree_repr [BOOL]: Representation of the input joint angle (Degree).
        Examples:
            self.forward_kinematics(0, [0.0, 45.0])
        """

        self.__theta_target = np.zeros(3)
        self.__theta_target[0] = theta[0]
        self.__theta_target[1] = theta[1]
        self.__theta_target[2] = theta[2]

        if degree_repr == True:
            self.rDH_param.theta = [x * (np.pi/180) for x in self.__theta_target]
        else:
            self.rDH_param.theta = self.__theta_target

        self.__fast_calc_fk()

    def inverse_kinematics(self, p, rot_deg, cfg):
        """
        Description:
            Inverse kinematics is the mathematical process of calculating the variable 
            joint parameters needed to place the end of a kinematic chain.
            Position of End-Effector (x, y) <-> Joint Angles (Theta_1, Theta_2, Theta_3)
        Args:
            (1) p [Float Array]: Position (x, y) of the target in meters.
            (2) rot_deg [INT]: Rotation of the end manipulator in degrees.
            (3) cfg [INT]: Robot configuration (IK Multiple Solutions).
        Examples:
            self.inverse_kinematics([0.45, 0.10], 30, 0)
        """
        rot_rad = np.pi * (rot_deg) / 180

        theta_aux = np.zeros(3)
        self.__p_target = np.zeros(2)
        self.__p_target[0] = p[0]
        self.__p_target[1] = p[1]

        p_temp = np.zeros(2)
        p_temp[0] = p[0] - np.cos(rot_rad) * self.rDH_param.a[2]
        p_temp[1] = p[1] - np.sin(rot_rad) * self.rDH_param.a[2]

        # Cosine Theorem [Beta]: eq (1)
        cosT_beta_numerator   = (self.rDH_param.a[0]**2 + p_temp[0]**2 + p_temp[1]**2 - self.rDH_param.a[1]**2)
        cosT_beta_denumerator = (2*self.rDH_param.a[0]*np.sqrt(p_temp[0]**2 + p_temp[1]**2))
        
        # Calculation angle of Theta 1,2 (Inverse trigonometric functions):
        # Rule 1: The range of the argument “x” for arccos function is limited from -1 to 1.
        # −1 ≤ x ≤ 1
        # Rule 2: Output of arccos is limited from 0 to π (radian).
        # 0 ≤ y ≤ π

        # Calculation angle of Theta 1
        if cosT_beta_numerator/cosT_beta_denumerator > 1:
            theta_aux[0] = np.arctan2(p_temp[1], p_temp[0]) 
            print('[INFO] Theta 1 Error: ', p_temp[1], p_temp[0])
        elif cosT_beta_numerator/cosT_beta_denumerator < -1:
            theta_aux[0] = np.arctan2(p_temp[1], p_temp[0]) - np.pi 
            print('[INFO] Theta 1 Error: ', p_temp[1], p_temp[0]) 
        else:
            if cfg == 0:
                theta_aux[0] = np.arctan2(p_temp[1], p_temp[0]) - np.arccos(cosT_beta_numerator/cosT_beta_denumerator)
            elif cfg == 1:
                theta_aux[0] = np.arctan2(p_temp[1], p_temp[0]) + np.arccos(cosT_beta_numerator/cosT_beta_denumerator)
                
        # Cosine Theorem [Alpha]: eq (2)
        cosT_alpha_numerator   = (self.rDH_param.a[0]**2) + (self.rDH_param.a[1]**2) - (p_temp[0]**2 + p_temp[1]**2)
        cosT_alpha_denumerator = (2*(self.rDH_param.a[0]*self.rDH_param.a[1]))

        # Calculation angle of Theta 2
        if cosT_alpha_numerator/cosT_alpha_denumerator > 1:
            theta_aux[1] = np.pi
            print('[INFO] Theta 2 Error: ', p_temp[1], p_temp[0])
        elif cosT_alpha_numerator/cosT_alpha_denumerator < -1:
            theta_aux[1] = 0.0
            print('[INFO] Theta 2 Error: ', p_temp[1], p_temp[0])
        else:
            if cfg == 0:
                theta_aux[1] = np.pi - np.arccos(cosT_alpha_numerator/cosT_alpha_denumerator)
            elif cfg == 1:
                theta_aux[1] = np.arccos(cosT_alpha_numerator/cosT_alpha_denumerator) - np.pi

        self.theta = theta_aux
        self.theta[1] = self.theta[1] + self.theta[0]
        self.theta[2] = rot_rad

        # Calculate the forward kinematics from the results of the inverse kinematics.
        self.forward_kinematics(self.theta, False)

    def _display_workspace(self, display_type = 0):
        """
        Description:
            Display the work envelope (workspace) in the environment.
        Args:
            (1) display_type [INT]: Work envelope visualization options (0: Mesh, 1: Points).
        Examples:
            self._display_workspace(0)
        """

        # Generate linearly spaced vectors for the each of joints.
        theta_1 = np.linspace((self.ax_wr[0][0]) * (np.pi/180), (self.ax_wr[0][1]) * (np.pi/180), 25)
        theta_2 = np.linspace((self.ax_wr[1][0]) * (np.pi/180), (self.ax_wr[1][1]) * (np.pi/180), 25)
        theta_3 = np.linspace((self.ax_wr[2][0]) * (np.pi/180), (self.ax_wr[2][1]) * (np.pi/180), 25)

        # Return coordinate matrices from coordinate vectors.
        [theta_1_mg, theta_2_mg, theta_3_mg] = np.meshgrid(theta_1, theta_2, theta_3)

        # Find the points x, y in the workspace using the equations FK.
        x_p =   (self.rDH_param.a[0]*np.cos(theta_1_mg) + self.rDH_param.a[1]*np.cos(theta_1_mg + theta_2_mg) + 
                self.rDH_param.a[2]*np.cos(theta_1_mg + theta_2_mg + theta_3_mg))
        y_p =   (self.rDH_param.a[0]*np.sin(theta_1_mg) + self.rDH_param.a[1]*np.sin(theta_1_mg + theta_2_mg) +
                self.rDH_param.a[2]*np.sin(theta_1_mg + theta_2_mg + theta_3_mg))

        x_p = np.reshape(x_p, (125, 125))
        y_p = np.reshape(y_p, (125, 125))

        if display_type == 0:
            plt.fill(x_p, y_p,'o', c=[0,1,0,0.01])
            plt.plot(x_p[0][0], y_p[0][0],'.', label=u"Work Envelop", c=[0,1,0,0.1])
        elif display_type == 1:
            plt.plot(x_p, y_p,'o', c=[0,1,0,0.1])
            plt.plot(x_p[0][0],y_p[0][0], '.', label=u"Work Envelop", c=[0,1,0,0.5])

    def display_environment(self, work_envelope = [False, 0]):
        """
        Description:
            Display the entire environment with the robot and other functions.
        Args:
            (1) work_envelope [Array (BOOL, INT)]: Work Envelop options (Visibility, Type of visualization (0: Mesh, 1: Points)).
        Examples:
            self.display_environment([True, 0])
        """

        # Display FK/IK calculation results (depens on type of options)
        self.__display_result()

        # Condition for visible work envelop
        if work_envelope[0] == True:
            self._display_workspace(work_envelope[1])

        if not (self.__p_target is None):
            if (self.__p_target[0] != self.p2[0]) and (self.__p_target[1] != self.p2[1]):
                plt.plot(self.__p_target[0], self.__p_target[1], label=r'Target Position: $p_{(x, y)}$', marker = 'o', ms = 30, mfc = [1,0,0], markeredgecolor = [0,0,0], mew = 5)
            else:
                plt.plot(self.__p_target[0], self.__p_target[1], label=r'Target Position: $p_{(x, y)}$', marker = 'o', ms = 30, mfc = [1,1,0], markeredgecolor = [0,0,0], mew = 5)

        # Visible -> Robot Base Recantgle
        plt.gca().add_patch(
            patches.Rectangle((-0.05, -0.05), 0.1, 0.1, label=r'Robot Base', facecolor=[0, 0, 0, 0.25]
            )
        )

        # Visible -> Arm 1: Joint 0 <-> Joint 1
        plt.plot([0.0, self.rDH_param.a[0]*np.cos(self.rDH_param.theta[0])],
            [0.0, self.rDH_param.a[0]*np.sin(self.rDH_param.theta[0])],
            'k-',linewidth=10
        )

        # Visible -> Arm 2: Joint 1 <-> Joint 2
        plt.plot([self.rDH_param.a[0]*np.cos(self.rDH_param.theta[0]), self.p1[0]],
            [self.rDH_param.a[0]*np.sin(self.rDH_param.theta[0]), self.p1[1]],
            'k-',linewidth=10
        )

        # Visible -> Arm 3: Joint 2 <-> End-Effector
        plt.plot([self.p1[0], self.p2[0]],
            [self.p1[1], self.p2[1]],
            'k-',linewidth=10
        )

        # Visible -> Joint 1
        plt.plot(0.0, 0.0, 
            label=r'Joint 1: $\theta_1 ('+ str(self.ax_wr[0][0]) +','+ str(self.ax_wr[0][1]) +')$', 
            marker = 'o', ms = 15, mfc = [1,1,1], markeredgecolor = [0,0,0], mew = 5
        )

        # Visible -> Joint 2
        plt.plot(self.rDH_param.a[0]*np.cos(self.rDH_param.theta[0]), self.rDH_param.a[0]*np.sin(self.rDH_param.theta[0]), 
            label=r'Joint 2: $\theta_2 ('+ str(self.ax_wr[1][0]) +','+ str(self.ax_wr[1][1]) +')$', 
            marker = 'o', ms = 15, mfc = [0.7, 0.0, 1, 1], markeredgecolor = [0,0,0], mew = 5
        )

        # Visible -> Joint 3
        plt.plot(self.p1[0], self.p1[1], 
            label=r'Joint 3: $\theta_3 ('+ str(self.ax_wr[2][0]) +','+ str(self.ax_wr[2][1]) +')$', 
            marker = 'o', ms = 15, mfc = [0.0, 0.7, 0.2, 1], markeredgecolor = [0,0,0], mew = 5
        )

        # Visible -> End-Effector
        plt.plot(self.p2[0], self.p2[1], 
            label=r'End-Effector Position: $EE_{(x, y)}$', 
            marker = 'o', ms = 15, mfc = [0,0.75,1, 1], markeredgecolor = [0,0,0], mew = 5
        )

        # Set minimum / maximum environment limits
        plt.axis([(-1)*(self.rDH_param.a[0] + self.rDH_param.a[1] + self.rDH_param.a[2]) - 0.2, 
            (1)*(self.rDH_param.a[0] + self.rDH_param.a[1] + self.rDH_param.a[2]) + 0.2, 
            (-1)*(self.rDH_param.a[0] + self.rDH_param.a[1] + self.rDH_param.a[2]) - 0.2, 
            (1)*(self.rDH_param.a[0] + self.rDH_param.a[1] + self.rDH_param.a[2]) + 0.2])

        # Set additional parameters for successful display of the robot environment
        plt.grid()
        plt.xlabel('x position [m]', fontsize = 20, fontweight ='normal')
        plt.ylabel('y position [m]', fontsize = 20, fontweight ='normal')
        plt.title(self.robot_name, fontsize = 50, fontweight ='normal')
        plt.legend(loc=0,fontsize=20)
        plt.show()

    def __display_rDHp(self):
        """
        Description: 
            Display the DH robot parameters.
        """

        print('[INFO] The Denavit-Hartenberg modified parameters of robot %s:' % (self.robot_name))
        print('[INFO] theta = [%f, %f, %f]' % (self.rDH_param.theta[0], self.rDH_param.theta[1], self.rDH_param.theta[2]))
        print('[INFO] a     = [%f, %f, %f]' % (self.rDH_param.a[0], self.rDH_param.a[1], self.rDH_param.a[2]))
        print('[INFO] d     = [%f, %f, %f]' % (self.rDH_param.d[0], self.rDH_param.d[1], self.rDH_param.d[2]))
        print('[INFO] alpha = [%f, %f, %f]' % (self.rDH_param.alpha[0], self.rDH_param.alpha[1], self.rDH_param.alpha[2]))

    def __display_result(self):
        """
        Description: 
            Display of the result of the kinematics forward/inverse of the robot and other parameters.
        """

        print('[INFO] Result of Kinematics calculation:')
        print('[INFO] Robot: %s' % (self.robot_name))

        if not (self.__p_target is None):
            print('[INFO] Target Position (End-Effector):')
            print('[INFO] p_t  = [x: %f, y: %f]' % (self.__p_target[0], self.__p_target[1]))
            print('[INFO] Target Position (Joint):')
            print('[INFO] Theta  = [Theta_1: %f, Theta_2: %f, Theta_3: %f]' % (self.__theta_target[0], self.__theta_target[1], self.__theta_target[2]))
            print('[INFO] Actual Position (End-Effector):')
            print('[INFO] p_ee = [x: %f, y: %f]' % (self.p2[0], self.p2[1]))
            print('[INFO] Actual Position (Joint):')
            print('[INFO] Theta  = [Theta_1: %f, Theta_2: %f, Theta_3: %f]' % (self.theta[0], self.theta[1], self.theta[2]))
        else:
            print('[INFO] Target Position (Joint):')
            print('[INFO] Theta  = [Theta_1: %f, Theta_2: %f, Theta_3: %f]' % (self.__theta_target[0], self.__theta_target[1], self.__theta_target[2]))
            print('[INFO] Actual Position (End-Effector):')
            print('[INFO] p_ee = [x: %f, y: %f]' % (self.p2[0], self.p2[1]))