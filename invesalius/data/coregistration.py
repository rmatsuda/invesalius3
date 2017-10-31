#--------------------------------------------------------------------------
# Software:     InVesalius - Software de Reconstrucao 3D de Imagens Medicas
# Copyright:    (C) 2001  Centro de Pesquisas Renato Archer
# Homepage:     http://www.softwarepublico.gov.br
# Contact:      invesalius@cti.gov.br
# License:      GNU - GPL 2 (LICENSE.txt/LICENCA.txt)
#--------------------------------------------------------------------------
#    Este programa e software livre; voce pode redistribui-lo e/ou
#    modifica-lo sob os termos da Licenca Publica Geral GNU, conforme
#    publicada pela Free Software Foundation; de acordo com a versao 2
#    da Licenca.
#
#    Este programa eh distribuido na expectativa de ser util, mas SEM
#    QUALQUER GARANTIA; sem mesmo a garantia implicita de
#    COMERCIALIZACAO ou de ADEQUACAO A QUALQUER PROPOSITO EM
#    PARTICULAR. Consulte a Licenca Publica Geral GNU para obter mais
#    detalhes.
#--------------------------------------------------------------------------

import threading
from time import sleep
from math import cos, sin

from numpy import asmatrix, mat, vstack, radians, identity
import wx
from wx.lib.pubsub import pub as Publisher

import invesalius.data.coordinates as dco
# import invesalius.data.bases as db
# import invesalius.data.transformations as tr

# TODO: Optimize navigation thread. Remove the infinite loop and optimize sleep.


class CoregistrationStatic(threading.Thread):
    """
    Thread to update the coordinates with the fiducial points
    co-registration method while the Navigation Button is pressed.
    Sleep function in run method is used to avoid blocking GUI and
    for better real-time navigation
    """

    def __init__(self, bases, nav_id, trck_info):
        threading.Thread.__init__(self)
        self.bases = bases
        self.nav_id = nav_id
        self.trck_info = trck_info
        self._pause_ = False
        self.start()
        
    def stop(self):
        self._pause_ = True
       
    def run(self):
        m_inv = self.bases[0]
        n = self.bases[1]
        q1 = self.bases[2]
        q2 = self.bases[3]
        trck_init = self.trck_info[0]
        trck_id = self.trck_info[1]
        trck_mode = self.trck_info[2]

        while self.nav_id:
            # trck_coord, probe, reference = dco.GetCoordinates(trck_init, trck_id, trck_mode)
            coord_raw = dco.GetCoordinates(trck_init, trck_id, trck_mode)

            trck_coord = coord_raw[0, :]

            trck_xyz = mat([[trck_coord[0]], [trck_coord[1]], [trck_coord[2]]])
            img = q1 + (m_inv*n)*(trck_xyz - q2)

            coord = (float(img[0]), float(img[1]), float(img[2]), trck_coord[3],
                     trck_coord[4], trck_coord[5])
            angles = coord_raw[0, 3:6]

            # Tried several combinations and different locations to send the messages,
            # however only this one does not block the GUI during navigation.
            wx.CallAfter(Publisher.sendMessage, 'Co-registered points', coord[0:3])
            wx.CallAfter(Publisher.sendMessage, 'Set camera in volume', coord)
            wx.CallAfter(Publisher.sendMessage, 'Update tracker angles', angles)

            # TODO: Optimize the value of sleep for each tracking device.
            # Debug tracker is not working with 0.175 so changed to 0.2
            # However, 0.2 is too low update frequency ~5 Hz. Need optimization URGENTLY.
            #sleep(.3)
            sleep(0.175)

            if self._pause_:
                return


class CoregistrationDynamic(threading.Thread):
    """
    Thread to update the coordinates with the fiducial points
    co-registration method while the Navigation Button is pressed.
    Sleep function in run method is used to avoid blocking GUI and
    for better real-time navigation
    """

    def __init__(self, bases, nav_id, trck_info):
        threading.Thread.__init__(self)
        self.bases = bases
        self.nav_id = nav_id
        self.trck_info = trck_info
        self._pause_ = False
        self.start()

    def stop(self):
        self._pause_ = True

    def run(self):
        m_inv = self.bases[0]
        n = self.bases[1]
        q1 = self.bases[2]
        q2 = self.bases[3]
        trck_init = self.trck_info[0]
        trck_id = self.trck_info[1]
        trck_mode = self.trck_info[2]

        while self.nav_id:
            # trck_coord, probe, reference = dco.GetCoordinates(trck_init, trck_id, trck_mode)
            coord_raw = dco.GetCoordinates(trck_init, trck_id, trck_mode)

            trck_coord = dco.dynamic_reference(coord_raw[0, :], coord_raw[1, :])

            trck_xyz = mat([[trck_coord[0]], [trck_coord[1]], [trck_coord[2]]])
            img = q1 + (m_inv * n) * (trck_xyz - q2)

            coord = (float(img[0]), float(img[1]), float(img[2]), trck_coord[3],
                     trck_coord[4], trck_coord[5])
            angles = coord_raw[0, 3:6]

            # Tried several combinations and different locations to send the messages,
            # however only this one does not block the GUI during navigation.
            wx.CallAfter(Publisher.sendMessage, 'Co-registered points', coord[0:3])
            wx.CallAfter(Publisher.sendMessage, 'Set camera in volume', coord)
            wx.CallAfter(Publisher.sendMessage, 'Update tracker angles', angles)

            # TODO: Optimize the value of sleep for each tracking device.
            # Debug tracker is not working with 0.175 so changed to 0.2
            # However, 0.2 is too low update frequency ~5 Hz. Need optimization URGENTLY.
            # sleep(.3)
            sleep(0.175)

            if self._pause_:
                return

class CoregistrationDynamic_m(threading.Thread):
    """
    Thread to update the coordinates with the fiducial points
    co-registration method while the Navigation Button is pressed.
    Sleep function in run method is used to avoid blocking GUI and
    for better real-time navigation
    """

    def __init__(self, bases, nav_id, trck_info):
        threading.Thread.__init__(self)
        self.bases = bases
        self.nav_id = nav_id
        self.trck_info = trck_info
        self._pause_ = False
        self.start()

    def stop(self):
        self._pause_ = True

    def run(self):
        m_change = self.bases

        trck_init = self.trck_info[0]
        trck_id = self.trck_info[1]
        trck_mode = self.trck_info[2]

        while self.nav_id:
            coord_raw = dco.GetCoordinates(trck_init, trck_id, trck_mode)

            trck_coord = dco.dynamic_reference_m(coord_raw[0, :], coord_raw[1, :])

            trck_coord_2 = vstack((asmatrix(trck_coord[0:3]).reshape([3, 1]), 1.))
            img = m_change * trck_coord_2

            coord = (float(img[0]), float(img[1]), float(img[2]), 0., 0., 0.)

            # Tried several combinations and different locations to send the messages,
            # however only this one does not block the GUI during navigation.
            wx.CallAfter(Publisher.sendMessage, 'Co-registered points', (None, coord))
            # wx.CallAfter(Publisher.sendMessage, 'Set camera in volume', coord)
            # wx.CallAfter(Publisher.sendMessage, 'Update tracker angles', angles)

            # TODO: Optimize the value of sleep for each tracking device.
            # Debug tracker is not working with 0.175 so changed to 0.2
            # However, 0.2 is too low update frequency ~5 Hz. Need optimization URGENTLY.
            # sleep(.3)
            sleep(0.175)

            if self._pause_:
                return


class CoregistrationObjectStatic(threading.Thread):
    """
    Thread to update the coordinates with the fiducial points
    co-registration method while the Navigation Button is pressed.
    Sleep function in run method is used to avoid blocking GUI and
    for better real-time navigation
    """

    def __init__(self, bases, nav_id, trck_info, coil_info):
        threading.Thread.__init__(self)
        self.bases = bases
        self.nav_id = nav_id
        self.trck_info = trck_info
        self.coil_info = coil_info
        self._pause_ = False
        self.start()

    def stop(self):
        self._pause_ = True

    def run(self):
        m_inv = self.bases[0]
        n = self.bases[1]
        q1 = self.bases[2]
        q2 = self.bases[3]
        trck_init = self.trck_info[0]
        trck_id = self.trck_info[1]
        trck_mode = self.trck_info[2]

        m_inv_trck = self.coil_info[0]
        obj_center_trck = self.coil_info[1]
        obj_sensor_trck = self.coil_info[2]
        obj_ref_id = self.coil_info[3]

        while self.nav_id:
            coord_raw = dco.GetCoordinates(trck_init, trck_id, trck_mode)

            if obj_ref_id:
                trck_xyz = asmatrix(coord_raw[2, 0:3]).reshape([3, 1]) + (obj_center_trck - obj_sensor_trck)
            else:
                trck_xyz = asmatrix(coord_raw[0, 0:3]).reshape([3, 1]) + (obj_center_trck - obj_sensor_trck)

            img = q1 + (m_inv * n) * (trck_xyz - q2)

            coord = (float(img[0]), float(img[1]), float(img[2]),
                     coord_raw[0, 3], coord_raw[0, 4], coord_raw[0, 5])
            angles = coord_raw[0, 3:6]

            # Tried several combinations and different locations to send the messages,
            # however only this one does not block the GUI during navigation.
            wx.CallAfter(Publisher.sendMessage, 'Co-registered points', coord[0:3])
            wx.CallAfter(Publisher.sendMessage, 'Set camera in volume', coord)
            wx.CallAfter(Publisher.sendMessage, 'Update object orientation',
                         (m_inv_trck, angles, coord[0:3]))
            wx.CallAfter(Publisher.sendMessage, 'Update tracker angles', angles)

            # TODO: Optimize the value of sleep for each tracking device.
            # Debug tracker is not working with 0.175 so changed to 0.2
            # However, 0.2 is too low update frequency ~5 Hz. Need optimization URGENTLY.
            # sleep(.3)
            sleep(0.175)

            if self._pause_:
                return


class CoregistrationObjectDynamic(threading.Thread):
    """
    Thread to update the coordinates with the fiducial points
    co-registration method while the Navigation Button is pressed.
    Sleep function in run method is used to avoid blocking GUI and
    for better real-time navigation
    """

    def __init__(self, bases, nav_id, trck_info, obj_data):
        threading.Thread.__init__(self)
        self.bases = bases
        self.nav_id = nav_id
        self.trck_info = trck_info
        self.obj_data = obj_data
        self._pause_ = False
        self.start()

    def stop(self):
        self._pause_ = True

    def run(self):
        m_inv = self.bases[0]
        n = self.bases[1]
        q1 = asmatrix(self.bases[2]).reshape([3, 1])
        q2 = asmatrix(self.bases[3]).reshape([3, 1])
        trck_init = self.trck_info[0]
        trck_id = self.trck_info[1]
        trck_mode = self.trck_info[2]

        # m_inv_trck = self.obj_data[0]
        # obj_center_trck = self.obj_data[1]
        # obj_sensor_trck = self.obj_data[2]
        # obj_ref_id = self.obj_data[3]

        obj_center_trck = self.obj_data[0]
        m_obj = self.obj_data[1]
        m_inv_obj = self.obj_data[2]
        r_obj = self.obj_data[3]
        fixed_sensor = self.obj_data[4]

        while self.nav_id:
            coord_raw = dco.GetCoordinates(trck_init, trck_id, trck_mode)

            # coord_raw[0, :3] += obj_center_trck

            trck_coord = asmatrix(dco.dynamic_reference(coord_raw[0, :], coord_raw[1, :])).reshape([6, 1])
            q_obj = asmatrix(dco.dynamic_reference(obj_center_trck, coord_raw[1, :])).reshape([6, 1])[:3, 0]
            q_fixed = asmatrix(dco.dynamic_reference(fixed_sensor, coord_raw[1, :])).reshape([6, 1])[:3, 0]
            # print asmatrix(dco.dynamic_reference(obj_center_trck, coord_raw[1, :])).reshape([6, 1])
            # print q_obj
            # q_obj = asmatrix(dco.dynamic_reference(obj_center_trck, coord_raw[1, :])).reshape([6, 1])[0, :3]

            p_dyn = trck_coord[:3, 0]

            img = q1 + (m_inv * n) * (p_dyn + (q_obj - q_fixed) - q2)
            # img = q1 + (m_inv * n) * (p_dyn - q2)

            a, b, g = radians(trck_coord[3, 0]), radians(trck_coord[4, 0]), radians(trck_coord[5, 0])

            m_rot = mat([[cos(a) * cos(b), sin(b) * sin(g) * cos(a) - cos(g) * sin(a),
                          cos(a) * sin(b) * cos(g) + sin(a) * sin(g)],
                         [cos(b) * sin(a), sin(b) * sin(g) * sin(a) + cos(g) * cos(a),
                          cos(g) * sin(b) * sin(a) - sin(g) * cos(a)],
                         [-sin(b), sin(g) * cos(b), cos(b) * cos(g)]])

            a, b, g = radians(coord_raw[1, 3:6])

            m_rot_sensor = mat([[cos(a) * cos(b), sin(b) * sin(g) * cos(a) - cos(g) * sin(a),
                          cos(a) * sin(b) * cos(g) + sin(a) * sin(g)],
                         [cos(b) * sin(a), sin(b) * sin(g) * sin(a) + cos(g) * cos(a),
                          cos(g) * sin(b) * sin(a) - sin(g) * cos(a)],
                         [-sin(b), sin(g) * cos(b), cos(b) * cos(g)]])

            # m_rot_obj = m_inv_obj*m_rot_sensor.T*m_rot*m_obj
            m_rot_obj = identity(3)


            coord = (float(img[0]), float(img[1]), float(img[2]), coord_raw[0, 3],
                     coord_raw[0, 4], coord_raw[0, 5])

            # Tried several combinations and different locations to send the messages,
            # however only this one does not block the GUI during navigation.
            # wx.CallAfter(Publisher.sendMessage, 'Co-registered points', coord[0:3])
            # wx.CallAfter(Publisher.sendMessage, 'Co-registered points', (m_inv_trck, coord))
            wx.CallAfter(Publisher.sendMessage, 'Co-registered points', (m_rot_obj, coord))
            # wx.CallAfter(Publisher.sendMessage, 'Set camera in volume', coord)
            # wx.CallAfter(Publisher.sendMessage, 'Update object orientation',
            #              (m_inv_trck, coord))
            # wx.CallAfter(Publisher.sendMessage, 'Update tracker angles', angles)

            # TODO: Optimize the value of sleep for each tracking device.
            # Debug tracker is not working with 0.175 so changed to 0.2
            # However, 0.2 is too low update frequency ~5 Hz. Need optimization URGENTLY.
            sleep(.3)
            # sleep(0.175)
            # sleep(1.)

            if self._pause_:
                return
