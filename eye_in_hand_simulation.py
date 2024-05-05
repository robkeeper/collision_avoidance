import pybullet as p
import pybullet_data
import numpy as np
import cv2
import math

from termcolor import cprint
import yellow_detect
import pose_estimation
from time import sleep

from pybullet_planning import BASE_LINK, RED, BLUE, GREEN
from pybullet_planning import load_pybullet, connect, wait_for_user, LockRenderer, has_gui, WorldSaver, HideOutput, \
    reset_simulation, disconnect, set_camera_pose, has_gui, set_camera, wait_for_duration, wait_if_gui, apply_alpha
from pybullet_planning import Pose, Point, Euler
from pybullet_planning import multiply, invert, get_distance
from pybullet_planning import create_obj, create_attachment, Attachment
from pybullet_planning import link_from_name, get_link_pose, get_moving_links, get_link_name, get_disabled_collisions, \
    get_body_body_disabled_collisions, has_link, are_links_adjacent
from pybullet_planning import get_num_joints, get_joint_names, get_movable_joints, set_joint_positions, joint_from_name, \
    joints_from_names, get_sample_fn, plan_joint_motion
from pybullet_planning import dump_world, set_pose
from pybullet_planning import get_collision_fn, get_floating_body_collision_fn, expand_links, create_box
from pybullet_planning import pairwise_collision, pairwise_collision_info, draw_collision_diagnosis, body_collision_info

#项目设置------------------------------------------------------------

# 连接引擎
_ = p.connect(p.GUI)
# 展示GUI的套件
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
# 添加资源路径
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0,0,-9.8)
time_step = 1./240. 
p.setTimeStep(time_step)

#物体生成------------------------------------------------------------

#为了生成物体，暂时关闭渲染
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

#载入外部urfd
planeUid = p.loadURDF("plane.urdf", useMaximalCoordinates=True)  # 加载一个地面
#trayUid = p.loadURDF("tray/traybox.urdf", basePosition=[0.4, 0, 0])  # 加载一个箱子，设置初始位置为（0，0，0）
pandaUid = p.loadURDF("franka_panda/panda.urdf",useFixedBase=True) #导入panda机器人

#生成不同区域
areaLength=0.5
areaWidth=0.3
areaHeight=0.001
rgbaList=[[1,1,0.4,1],[0,1,0,1],[1,0.2,0,1]]
startAreaCollisionId = p.createCollisionShape(p.GEOM_BOX,
                                  halfExtents=[areaWidth,areaLength, areaHeight])
collectAreaCollisionId = p.createCollisionShape(p.GEOM_BOX,
                                  halfExtents=[areaLength, areaWidth,areaHeight])
startAreaVisualId = p.createVisualShape(p.GEOM_BOX,
                                  halfExtents=[areaWidth, areaLength,areaHeight],
                                  rgbaColor=[0,0,0,1])
p.createMultiBody(baseMass=10,
                    baseCollisionShapeIndex=startAreaCollisionId,
                    baseVisualShapeIndex=startAreaVisualId,
                    basePosition=[0.4,0,0])
for i in range(3):
    p.createMultiBody(baseMass=10,
                    baseCollisionShapeIndex=collectAreaCollisionId,
                    baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX,
                                  halfExtents=[areaLength, areaWidth,areaHeight],
                                  rgbaColor=rgbaList[i]),
                    basePosition=[-0.7,-0.7+i*0.7,0])




#生成自定义彩色盒子尺寸
boxHalfLength = 0.0132
boxHalfWidth = 0.0132
boxHalfHeight = 0.02
#盒子的碰撞体形状
BoxCollisionId = p.createCollisionShape(p.GEOM_BOX,
                                  halfExtents=[boxHalfLength, boxHalfWidth, boxHalfHeight])
#盒子的三种颜色视觉形状
yellowBoxId = p.createVisualShape(p.GEOM_BOX,
                                  halfExtents=[boxHalfLength, boxHalfWidth, boxHalfHeight],
                                  rgbaColor=[1,1,0.4,1])
greenBoxId = p.createVisualShape(p.GEOM_BOX,
                                  halfExtents=[boxHalfLength, boxHalfWidth, boxHalfHeight],
                                  rgbaColor=[0,1,0,1])
redBoxId = p.createVisualShape(p.GEOM_BOX,
                                  halfExtents=[boxHalfLength, boxHalfWidth, boxHalfHeight],
                                  rgbaColor=[1,0.2,0,1])
pointId=p.createVisualShape(p.GEOM_BOX,
                                  halfExtents=[0.005,0.005,0.005],
                                  rgbaColor=[0,0,1,1])
#生成物体
pos1=[0.3, 0, 0.02]
pos2=[0.38, 0.08, 0.02]
pos3=[0.4, -0.16, 0.02]
p.createMultiBody(baseMass=0.5,
                    baseCollisionShapeIndex=BoxCollisionId,
                    baseVisualShapeIndex=yellowBoxId,
                    basePosition=pos1)
p.createMultiBody(baseMass=0.5,
                    baseCollisionShapeIndex=BoxCollisionId,
                    baseVisualShapeIndex=greenBoxId,
                    basePosition=pos2)
p.createMultiBody(baseMass=0.5,
                    baseCollisionShapeIndex=BoxCollisionId,
                    baseVisualShapeIndex=redBoxId,
                    basePosition=pos3)

#生成障碍物
block = create_box(0.05, 1, 2)
block_x = -0.75
block_y = -0.35
block_z = 0
set_pose(block, Pose(Point(x=block_x, y=block_y, z=block_z), Euler(yaw=np.pi/2)))

#重新开启渲染
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

#初始设置------------------------------------------------------------

#回到初始位置，松开抓夹
rest_poses = [0,-math.pi/3,0,-math.pi*3/4,0,math.pi/2,math.pi/4]
for i in range(7):
    p.resetJointState(pandaUid,i,rest_poses[i])
p.resetJointState(pandaUid,9,0.06)
p.resetJointState(pandaUid,10,0.06)

width = 1080  # 图像宽度
height = 720   # 图像高度

fov = 50  # 相机视角
aspect = width / height  # 宽高比
near = 0.01  # 最近拍摄距离
far = 20  # 最远拍摄距离

# 物体的世界坐标
width_box = 32
height_box = 32
obj_left_up = (-width_box/2,-height_box/2,0)
obj_right_up = (width_box/2,-height_box/2,0)
obj_right_down = (width_box/2,height_box/2,0)
obj_left_down = (-width_box/2,height_box/2,0)
objpoints = np.array([obj_left_up,obj_right_up,obj_right_down,obj_left_down],dtype=np.double)
pos=[-0.2,-1.5,0.5]

sleep(1)
color_detect_flag= 0 #限制颜色识别运行次数.0:未识别 1:识别完成
state_flag=0 #识别进程 0:未完成移动,1:已完成运动,未夹取,2:已夹取3:移动
finish_move_count=0 #完成移动的关节数量 一旦有未完成关节则清0
finish_grisp_time=0 #完成夹取的时间

#开始模拟

while True:
    joint_poses = p.calculateInverseKinematics(pandaUid,11,pos)
    for i in range(9):
        p.setJointMotorControl2(bodyIndex=pandaUid,
                                jointIndex=i,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=joint_poses[i],
                                maxVelocity=2)
    current_joint_position=p.getJointStates(pandaUid,list(range(9)))
    for i in range(9):
        if (abs(current_joint_position[i][0]-joint_poses[i]))<0.05:
            finish_move_count+=1
        else:
            finish_move_count=0
    if finish_move_count>=9 :
        state_flag=4
    else:
        finish_move_count=0
    # joint_poses = p.calculateInverseKinematics(pandaUid,11,pos)
    # print(len(joint_poses))
    # print(joint_poses)
    # arm_joints=range(9)
    # path = plan_joint_motion(pandaUid, arm_joints[0:7], joint_poses[0:7], obstacles=[block], self_collisions=False, custom_limits={arm_joints[1]:[-3, 3],arm_joints[7]:[-3, 3]})
    # if path is None:
    #     cprint('no plan found', 'red')
    #     finish_move_count=0
    # else:
    #     state_flag=4
    
    cv2.waitKey(1)

    sleep(time_step)
    p.stepSimulation()

