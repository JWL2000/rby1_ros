import numpy as np
from rby1_sdk.dynamics import Robot, load_robot_from_urdf
import rby1_sdk as rby


RBY1_LINK_NAMES: list[str] = ['base',
                                'wheel_l', 'wheel_r',
                                'link_torso_0', 'link_torso_1', 'link_torso_2', 'link_torso_3', 'link_torso_4', 'link_torso_5',
                                'link_head_1', 'link_left_arm_0', 'link_right_arm_0', 'link_head_2',
                                'link_left_arm_1', 'link_right_arm_1',
                                'link_left_arm_2', 'link_right_arm_2',
                                'link_left_arm_3', 'link_right_arm_3',
                                'link_left_arm_4', 'link_right_arm_4',
                                'link_left_arm_5', 'link_right_arm_5',
                                'link_left_arm_6', 'link_right_arm_6']

RBY1_JOINT_NAMES: list[str] = ['left_wheel', 'right_wheel',
                                'torso_0', 'torso_1', 'torso_2', 'torso_3', 'torso_4', 'torso_5',
                                'right_arm_0', 'right_arm_1', 'right_arm_2', 'right_arm_3',
                                'right_arm_4', 'right_arm_5', 'right_arm_6',
                                'left_arm_0', 'left_arm_1', 'left_arm_2', 'left_arm_3',
                                'left_arm_4', 'left_arm_5', 'left_arm_6',
                                'head_0', 'head_1']

class RBY1Dyn:
    def __init__(self):
        self.robot = load_robot_from_urdf("/home/choiyj/rby1-sdk/models/rby1a/urdf/model_v1.0.urdf", base_link_name="base")
        self.dyn_robot = Robot(self.robot)
        self.links: list[str] = ["base", "link_torso_5", "link_right_arm_6", "link_left_arm_6"]
        self.state = self.dyn_robot.make_state(self.links, RBY1_JOINT_NAMES)
        robot_max_q = self.dyn_robot.get_limit_q_upper(self.state)
        robot_min_q = self.dyn_robot.get_limit_q_lower(self.state)
        robot_max_qdot = self.dyn_robot.get_limit_qdot_upper(self.state)
        robot_max_qddot = self.dyn_robot.get_limit_qddot_upper(self.state)
        print("Robot max q:", robot_max_q)
        print("Robot min q:", robot_min_q)
        print("Robot max qdot:", robot_max_qdot)
        print("Robot max qddot:", robot_max_qddot)


    def get_fk(self, joint_positions):
        self.state.set_q(joint_positions)
        self.state.set_qdot(np.zeros_like(joint_positions))
        self.dyn_robot.compute_forward_kinematics(self.state)
        self.dyn_robot.compute_diff_forward_kinematics(self.state)
        fk_results = {}
        for link_idx in range(1, 4):
            link_name = self.links[link_idx]
            fk_results[link_name] = self.dyn_robot.compute_transformation(self.state, 0, link_idx)
        return fk_results
    
    def get_jacobian(self):
        J = self.dyn_robot.compute_body_jacobian(self.state, 0, 2)
        return J
    
    def get_ik(self, ee_idx, target_T, initial_q, max_iters=100, tol=1e-6) -> np.ndarray:
        '''
        ee_idx: 0-base, 1-torso, 2-right_arm, 3-left_arm
        '''
        q = np.array(initial_q, dtype=float).copy()

        lam = 1e-3
        
        max_step = 0.1
        for _ in range(max_iters):
            self.state.set_q(q)
            self.state.set_qdot(np.zeros_like(q))
            self.dyn_robot.compute_forward_kinematics(self.state)
            self.dyn_robot.compute_diff_forward_kinematics(self.state)

            # 현재 EE pose (base -> link_right_arm_6, index 2)
            current_T = self.dyn_robot.compute_transformation(self.state, 0, ee_idx)

            # 위치/자세 error 계산
            p_cur = current_T[0:3, 3]
            R_cur = current_T[0:3, 0:3]

            p_tar = target_T[0:3, 3]
            R_tar = target_T[0:3, 0:3]

            # position error
            e_p = p_tar - p_cur

            # orientation error
            R_err = R_cur.T @ R_tar
            e_o = 0.5 * np.array([
                R_err[2, 1] - R_err[1, 2],
                R_err[0, 2] - R_err[2, 0],
                R_err[1, 0] - R_err[0, 1]
            ])

            # 6x1 pose error
            e = np.concatenate((e_p, e_o))

            if np.linalg.norm(e) < tol:
                break

            # Jacobian (body jacobian)
            J = self.dyn_robot.compute_body_jacobian(self.state, 1, ee_idx)
            J = np.array(J, dtype=float)

            # shape 정리: 6 x N 이라고 가정, 아니면 transpose
            if J.shape[0] != 6 and J.shape[1] == 6:
                J = J.T

            # Damped Least Squares
            JT = J.T
            A = J @ JT + (lam ** 2) * np.eye(6)
            dq = JT @ np.linalg.solve(A, e)

            dq = np.clip(dq, -max_step, max_step)

            q = q + dq

            if hasattr(self, "robot_min_q") and hasattr(self, "robot_max_q"):
                q = np.minimum(np.maximum(q, self.robot_min_q), self.robot_max_q)

        return q

if __name__ == "__main__":
    print(rby.Model_A().robot_joint_names)
    rby1_dyn = RBY1Dyn()
    # joint_positions = np.deg2rad([0.0, 0.0] +
    #             [0.0, 45.0, -90.0, 45.0, 0.0, 0.0] +
    #             [0.0, -15.0, 0.0, -120.0, 0.0, 70.0, 0.0] +
    #             [0.0, 15.0, 0.0, -120.0, 0.0, 70.0, 0.0])
    joint_positions = np.deg2rad([0.0]*24)
    fk_results = rby1_dyn.get_fk(joint_positions)
    for link_name, transform in fk_results.items():
        print(f"Link: {link_name}")
        print(transform)

    J = rby1_dyn.get_jacobian()
    print("Jacobian:\n", J.T)