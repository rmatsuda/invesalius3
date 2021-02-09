import cv2
import cv2.aruco as aruco
import dlib
import numpy as np
from imutils import face_utils
import invesalius.data.stabilizer as stabilizer

import threading
import wx
from pubsub import pub as Publisher
from time import sleep

class camera(threading.Thread):

    def __init__(self, camera_queue, event):
        """Class (threading) to compute real time tractography data for visualization in a single loop.

        Different than ComputeTractsThread because it does not keep adding tracts to the bundle until maximum,
        is reached. It actually compute all requested tracts at once. (Might be deleted in the future)!
        Tracts are computed using the Trekker library by Baran Aydogan (https://dmritrekker.github.io/)
        For VTK visualization, each tract (fiber) is a constructed as a tube and many tubes combined in one
        vtkMultiBlockDataSet named as a branch. Several branches are combined in another vtkMultiBlockDataSet named as
        bundle, to obtain fast computation and visualization. The bundle dataset is mapped to a single vtkActor.
        Mapper and Actor are computer in the data/viewer_volume.py module for easier handling in the invesalius 3D scene.

        Sleep function in run method is used to avoid blocking GUI and more fluent, real-time navigation

        :param inp: List of inputs: trekker instance, affine numpy array, seed_offset, seed_radius, n_threads
        :type inp: list
        :param affine_vtk: Affine matrix in vtkMatrix4x4 instance to update objects position in 3D scene
        :type affine_vtk: vtkMatrix4x4
        :param coord_queue: Queue instance that manage coordinates read from tracking device and coregistered
        :type coord_queue: queue.Queue
        :param visualization_queue: Queue instance that manage coordinates to be visualized
        :type visualization_queue: queue.Queue
        :param event: Threading event to coordinate when tasks as done and allow UI release
        :type event: threading.Event
        :param sle: Sleep pause in seconds
        :type sle: float
        """

        threading.Thread.__init__(self, name='Camera')

        self.camera_init = None

        self.camera_queue = camera_queue
        self.event = event

        self.face_landmark_path = 'D:\\Repository\\camera_tracking\\video_test_shape\\shape_predictor_68_face_landmarks.dat'

        K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
             0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
             0.0, 0.0, 1.0]
        #K = [1.74033827e+03, 0.00000000e+00, 5.83808276e+02,
        # 0.00000000e+00, 2.12379163e+03, 4.64720236e+02,
        # 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]
        #D = [-8.09614536e-03,  4.66000740e+00, -1.81727801e-02, -4.32406078e-03,  -2.42935822e+02]

        self.cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
        self.dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

        self.object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                                 [1.330353, 7.122144, 6.903745],
                                 [-1.330353, 7.122144, 6.903745],
                                 [-6.825897, 6.760612, 4.402142],
                                 [5.311432, 5.485328, 3.987654],
                                 [1.789930, 5.393625, 4.413414],
                                 [-1.789930, 5.393625, 4.413414],
                                 [-5.311432, 5.485328, 3.987654],
                                 [2.005628, 1.409845, 6.165652],
                                 [-2.005628, 1.409845, 6.165652],
                                 [2.774015, -2.080775, 5.048531],
                                 [-2.774015, -2.080775, 5.048531],
                                 [0.000000, -3.116408, 6.097667],
                                 [0.000000, -7.415691, 4.070434]])

        #Aruco parameters:
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters_create()
        self.parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self.parameters.cornerRefinementWinSize = 5

        #translate_tooltip = np.array([0, 0.21, 0])
        #self.translate_tooltip = np.array([0.059, 0.241, -0.005])
        self.translate_tooltip = np.array([0.040, 0.251, -0.005])

        markerLength = 0.03  #unit is meters.
        markerSeparation = 0.005  #unit is meters.
        #TODO: improve firstMarker numbers
        self.board_probe = aruco.GridBoard_create(2, 2, markerLength, markerSeparation, self.aruco_dict, firstMarker = 0)
        self.board_coil = aruco.GridBoard_create(2, 1, markerLength, markerSeparation, self.aruco_dict, firstMarker = 4)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Unable to connect to camera.")
            return
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.face_landmark_path)

        self.ref = np.zeros(6)
        self.probe = np.zeros(6)
        self.coil = np.zeros(6)

        self.face_window = stabilizer.SavGol(5)
        self.pose_window = stabilizer.SavGol(5)
        self.coil_window = stabilizer.SavGol(5)

        self.cap.read()

        print("Initialization OK");

    def run(self):

        while not self.event.is_set():
            _, frame = self.cap.read()
            # frame = imutils.resize(frame, width=400)

            # operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_rects = self.detector(gray, 0)

            # lists of ids and the corners beloning to each id
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
            aruco.refineDetectedMarkers(gray, self.board_probe, corners, ids, rejectedImgPoints)
            aruco.refineDetectedMarkers(gray, self.board_coil, corners, ids, rejectedImgPoints)

            if len(face_rects) > 0:
                shape = self.predictor(frame, face_rects[0])
                shape = face_utils.shape_to_np(shape)

                euler_angle, translation_vec = self.get_head_pose(shape)

                # The Savitzky-Golay transformations and filter for head pose
                euler_angle, translation_vec, self.face_window.array = stabilizer.SavGol.filter(
                    self.face_window.window,
                    self.face_window.array,
                    euler_angle,
                    translation_vec)

                angles = np.array([euler_angle[2], euler_angle[1], euler_angle[0]])
                self.ref = np.hstack([-10 * translation_vec[:, 0], angles[:, 0]])

                ref_id = 1
            else:
                ref_id = 0

            if np.all(ids != None):
                if np.any(ids == 0) or np.any(ids == 1) or np.any(ids == 2) or np.any(ids == 3):
                    retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, self.board_probe, self.cam_matrix,
                                                                 self.dist_coeffs, rvec=None, tvec=None)
                    tvec = np.float32(np.vstack(tvec))
                    # calc for probe board
                    rotation_mat = np.identity(4).astype(np.float32)
                    rotation_mat[:3, :3], _ = cv2.Rodrigues(rvec)
                    pose_mat = cv2.hconcat((rotation_mat[:3, :3], tvec))
                    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

                    tool_tip_position = np.dot(rotation_mat[:3, :3],
                                               np.transpose(self.translate_tooltip)) + np.transpose(tvec)
                    tooltip_mat = cv2.hconcat((rotation_mat[:3, :3], np.transpose(np.float32(tool_tip_position))))
                    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(tooltip_mat[:3, :])
                    tool_tip_position = np.reshape(tool_tip_position, (3, 1))

                    # Stabilize the pose.
                    # pose = (euler_angle, tvec)
                    # stabile_pose = []
                    # pose_np = np.array(pose).flatten()
                    # for value, ps_stb in zip(pose_np, self.probe_stabilizers):
                    #     ps_stb.update([value])
                    #     stabile_pose.append(ps_stb.state[0])
                    # stabile_pose = np.reshape(stabile_pose, (-1, 3))
                    # euler_angle, tvec = stabile_pose
                    # euler_angle = np.reshape(euler_angle, (3, 1))

                    ######
                    # The Savitzky-Golay transformations and filter for probe pose
                    euler_angle, tool_tip_position, self.pose_window.array = stabilizer.SavGol.filter(
                        self.pose_window.window,
                        self.pose_window.array,
                        euler_angle,
                        tool_tip_position)

                    angles = np.array([euler_angle[2], euler_angle[1], euler_angle[0]])
                    tool_tip_position = np.hstack(tool_tip_position)
                    self.probe = np.hstack([1000 * tool_tip_position, angles[:, 0]])


                elif np.any(ids == 4) or np.any(ids == 5):
                    retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, self.board_coil, self.cam_matrix,
                                                                 self.dist_coeffs, rvec=None, tvec=None)
                    tvec = np.vstack(tvec)
                    # calc for coil board
                    rotation_mat, _ = cv2.Rodrigues(rvec)
                    pose_mat = cv2.hconcat((rotation_mat, tvec))
                    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

                    # Stabilize the pose.
                    pose = (euler_angle, tvec)
                    stabile_pose = []
                    pose_np = np.array(pose).flatten()
                    for value, ps_stb in zip(pose_np, self.obj_stabilizers):
                        ps_stb.update([value])
                        stabile_pose.append(ps_stb.state[0])
                    stabile_pose = np.reshape(stabile_pose, (-1, 3))
                    euler_angle, tvec = stabile_pose
                    euler_angle = np.reshape(euler_angle, (3, 1))

                    angles = np.array([euler_angle[2], euler_angle[1], euler_angle[0]])

                    self.coil = np.hstack([1000 * tvec, angles[:, 0]])

                probe_id = 1
            else:
                probe_id = 0
            #cv2.imshow("demo", frame)
            # print(self.ref)
            wx.CallAfter(Publisher.sendMessage, 'Send camera coord',
                         coord=(np.vstack([self.probe, self.ref, self.coil]), probe_id, ref_id))
            sleep(0.1)

        else:
            None
            #if self.trigger_init:
             #   self.trigger_init.close()

    def Close(self):
        #self.cap.release()
        cv2.destroyAllWindows()
        self.cap.stop()
        self.cap.release()

    def get_head_pose(self, shape):
        image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                                shape[39], shape[42], shape[45], shape[31], shape[35],
                                shape[48], shape[54], shape[57], shape[8]])

        _, rotation_vec, translation_vec = cv2.solvePnP(self.object_pts, image_pts, self.cam_matrix, self.dist_coeffs,
                                                        flags=cv2.SOLVEPNP_ITERATIVE)

        # reprojectdst, _ = cv2.projectPoints(self.reprojectsrc, rotation_vec, translation_vec,self.cam_matrix,
        #                                     self.dist_coeffs)
        #
        # reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

        # calc euler angle
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

        # Stabilize the pose.
        # pose = (euler_angle, translation_vec)
        # stabile_pose = []
        # pose_np = np.array(pose).flatten()
        # for value, ps_stb in zip(pose_np, self.ref_stabilizers):
        #     ps_stb.update([value])
        #     stabile_pose.append(ps_stb.state[0])
        # stabile_pose = np.reshape(stabile_pose, (-1, 3))
        #euler_angle, translation_vec = stabile_pose
        euler_angle = np.reshape(euler_angle, (3, 1))
        translation_vec = np.reshape(translation_vec, (3, 1))

        return euler_angle, translation_vec


# class camera():
#     def __init__(self):
#         self.face_landmark_path = 'D:\\Repository\\camera_tracking\\video_test_shape\\shape_predictor_68_face_landmarks.dat'
#
#         K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
#              0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
#              0.0, 0.0, 1.0]
#         #K = [1.74033827e+03, 0.00000000e+00, 5.83808276e+02,
#         # 0.00000000e+00, 2.12379163e+03, 4.64720236e+02,
#         # 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
#         D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]
#         #D = [-8.09614536e-03,  4.66000740e+00, -1.81727801e-02, -4.32406078e-03,  -2.42935822e+02]
#
#         self.cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
#         self.dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)
#
#         self.object_pts = np.float32([[6.825897, 6.760612, 4.402142],
#                                  [1.330353, 7.122144, 6.903745],
#                                  [-1.330353, 7.122144, 6.903745],
#                                  [-6.825897, 6.760612, 4.402142],
#                                  [5.311432, 5.485328, 3.987654],
#                                  [1.789930, 5.393625, 4.413414],
#                                  [-1.789930, 5.393625, 4.413414],
#                                  [-5.311432, 5.485328, 3.987654],
#                                  [2.005628, 1.409845, 6.165652],
#                                  [-2.005628, 1.409845, 6.165652],
#                                  [2.774015, -2.080775, 5.048531],
#                                  [-2.774015, -2.080775, 5.048531],
#                                  [0.000000, -3.116408, 6.097667],
#                                  [0.000000, -7.415691, 4.070434]])
#
#         #Aruco parameters:
#         self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
#         self.parameters = aruco.DetectorParameters_create()
#         self.parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
#         self.parameters.cornerRefinementWinSize = 5
#
#         #translate_tooltip = np.array([0, 0.21, 0])
#         self.translate_tooltip = np.array([0.059, 0.241, -0.005])
#
#         markerLength = 0.05  #unit is meters.
#         markerSeparation = 0.005  #unit is meters.
#         #TODO: improve firstMarker numbers
#         self.board_probe = aruco.GridBoard_create(2, 1, markerLength, markerSeparation, self.aruco_dict, firstMarker = 0)
#         self.board_coil = aruco.GridBoard_create(2, 1, markerLength, markerSeparation, self.aruco_dict, firstMarker = 2)
#
#         # self.ref_stabilizers = [stabilizer.Stabilizer_ref(
#         #     state_num=2,
#         #     measure_num=1,
#         #     cov_process=0.1,
#         #     cov_measure=1) for _ in range(6)]
#         # self.probe_stabilizers = [stabilizer.Stabilizer_probe(
#         #     state_num=2,
#         #     measure_num=1,
#         #     cov_process=0.1,
#         #     cov_measure=1) for _ in range(6)]
#         # self.obj_stabilizers = [stabilizer.Stabilizer_obj(
#         #     state_num=2,
#         #     measure_num=1,
#         #     cov_process=0.1,
#         #     cov_measure=1) for _ in range(6)]
#
#
#     def Initialize(self):
#         self.cap = cv2.VideoCapture(0)
#         if not self.cap.isOpened():
#             print("Unable to connect to camera.")
#             return
#         self.detector = dlib.get_frontal_face_detector()
#         self.predictor = dlib.shape_predictor(self.face_landmark_path)
#
#         self.ref = np.zeros(6)
#         self.probe = np.zeros(6)
#         self.coil = np.zeros(6)
#
#         self.face_window = stabilizer.SavGol(7)
#         self.pose_window = stabilizer.SavGol(7)
#         self.coil_window = stabilizer.SavGol(7)
#
#         self.cap.read()
#
#         print("Initialization OK");
#
#
#     def Run(self):
#         _, frame = self.cap.read()
#         #frame = imutils.resize(frame, width=400)
#
#         # operations on the frame come here
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         face_rects = self.detector(gray, 0)
#
#
#         # lists of ids and the corners beloning to each id
#         corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
#         aruco.refineDetectedMarkers(gray, self.board_probe, corners, ids, rejectedImgPoints)
#         aruco.refineDetectedMarkers(gray, self.board_coil, corners, ids, rejectedImgPoints)
#
#         if len(face_rects) > 0:
#             shape = self.predictor(frame, face_rects[0])
#             shape = face_utils.shape_to_np(shape)
#
#             euler_angle, translation_vec = self.get_head_pose(shape)
#
#             # The Savitzky-Golay transformations and filter for head pose
#             # euler_angle, translation_vec, self.face_window.array = stabilizer.SavGol.filter(
#             #     self.face_window.window,
#             #     self.face_window.array,
#             #     euler_angle,
#             #     translation_vec)
#
#             angles = np.array([euler_angle[2], euler_angle[1], euler_angle[0]])
#             self.ref = np.hstack([-10*translation_vec[:, 0], angles[:, 0]])
#
#             ref_id = 1
#         else:
#             ref_id = 0
#
#         if np.all(ids != None):
#             if np.any(ids == 0) or np.any(ids == 1):
#                 retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, self.board_probe, self.cam_matrix,
#                                                              self.dist_coeffs, rvec=None, tvec=None)
#                 tvec = np.float32(np.vstack(tvec))
#                 # calc for probe board
#                 rotation_mat = np.identity(4).astype(np.float32)
#                 rotation_mat[:3, :3], _ = cv2.Rodrigues(rvec)
#                 pose_mat = cv2.hconcat((rotation_mat[:3, :3], tvec))
#                 _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
#
#                 tool_tip_position = np.dot(rotation_mat[:3, :3],
#                                            np.transpose(self.translate_tooltip)) + np.transpose(tvec)
#                 tooltip_mat = cv2.hconcat((rotation_mat[:3, :3], np.transpose(np.float32(tool_tip_position))))
#                 _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(tooltip_mat[:3, :])
#                 tool_tip_position = np.reshape(tool_tip_position, (3, 1))
#
#                 # Stabilize the pose.
#                 # pose = (euler_angle, tvec)
#                 # stabile_pose = []
#                 # pose_np = np.array(pose).flatten()
#                 # for value, ps_stb in zip(pose_np, self.probe_stabilizers):
#                 #     ps_stb.update([value])
#                 #     stabile_pose.append(ps_stb.state[0])
#                 # stabile_pose = np.reshape(stabile_pose, (-1, 3))
#                 # euler_angle, tvec = stabile_pose
#                 # euler_angle = np.reshape(euler_angle, (3, 1))
#
#
#             ######
#                 # The Savitzky-Golay transformations and filter for probe pose
#                 # euler_angle, tool_tip_position, self.pose_window.array = stabilizer.SavGol.filter(
#                 #     self.pose_window.window,
#                 #     self.pose_window.array,
#                 #     euler_angle,
#                 #     tool_tip_position)
#
#                 angles = np.array([euler_angle[2], euler_angle[1], euler_angle[0]])
#                 tool_tip_position = np.hstack(tool_tip_position)
#                 self.probe = np.hstack([1000*tool_tip_position, angles[:, 0]])
#
#
#             elif np.any(ids == 2) or np.any(ids == 3):
#                 retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, self.board_coil, self.cam_matrix,
#                                                              self.dist_coeffs, rvec=None, tvec=None)
#                 tvec = np.vstack(tvec)
#                 # calc for coil board
#                 rotation_mat, _ = cv2.Rodrigues(rvec)
#                 pose_mat = cv2.hconcat((rotation_mat, tvec))
#                 _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
#
#                 # Stabilize the pose.
#                 pose = (euler_angle, tvec)
#                 stabile_pose = []
#                 pose_np = np.array(pose).flatten()
#                 for value, ps_stb in zip(pose_np, self.obj_stabilizers):
#                     ps_stb.update([value])
#                     stabile_pose.append(ps_stb.state[0])
#                 stabile_pose = np.reshape(stabile_pose, (-1, 3))
#                 euler_angle, tvec = stabile_pose
#                 euler_angle = np.reshape(euler_angle, (3, 1))
#
#                 angles = np.array([euler_angle[2], euler_angle[1], euler_angle[0]])
#
#                 self.coil = np.hstack([1000*tvec, angles[:, 0]])
#
#             probe_id = 1
#         else:
#             probe_id = 0
#         #cv2.imshow("demo", frame)
#         return np.vstack([self.probe, self.ref, self.coil]), probe_id, ref_id
#
#     def Close(self):
#         #self.cap.release()
#         cv2.destroyAllWindows()
#         self.cap.stop()
#         self.cap.release()
#
#     def get_head_pose(self, shape):
#         image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
#                                 shape[39], shape[42], shape[45], shape[31], shape[35],
#                                 shape[48], shape[54], shape[57], shape[8]])
#
#         _, rotation_vec, translation_vec = cv2.solvePnP(self.object_pts, image_pts, self.cam_matrix, self.dist_coeffs,
#                                                         flags=cv2.SOLVEPNP_ITERATIVE)
#
#         # reprojectdst, _ = cv2.projectPoints(self.reprojectsrc, rotation_vec, translation_vec,self.cam_matrix,
#         #                                     self.dist_coeffs)
#         #
#         # reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))
#
#         # calc euler angle
#         rotation_mat, _ = cv2.Rodrigues(rotation_vec)
#         pose_mat = cv2.hconcat((rotation_mat, translation_vec))
#         _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
#
#         # Stabilize the pose.
#         # pose = (euler_angle, translation_vec)
#         # stabile_pose = []
#         # pose_np = np.array(pose).flatten()
#         # for value, ps_stb in zip(pose_np, self.ref_stabilizers):
#         #     ps_stb.update([value])
#         #     stabile_pose.append(ps_stb.state[0])
#         # stabile_pose = np.reshape(stabile_pose, (-1, 3))
#         #euler_angle, translation_vec = stabile_pose
#         euler_angle = np.reshape(euler_angle, (3, 1))
#         translation_vec = np.reshape(translation_vec, (3, 1))
#
#         return euler_angle, translation_vec
#
#
