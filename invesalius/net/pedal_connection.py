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

import time
from threading import Thread

import mido

from invesalius import constants
from invesalius.pubsub import pub as Publisher
from invesalius.utils import Singleton

class PedalConnection(Thread, metaclass=Singleton):
    """
    Connect to the trigger pedal via MIDI, and allow adding callbacks for the pedal
    being pressed or released.

    Started by calling PedalConnection().start()
    """
    def __init__(self):
        Thread.__init__(self)
        self.daemon = True

        self.in_use = False

        self._midi_in = None
        self._active_inputs = None
        self._callback_infos = []

        self.serial = True
        self.com_port = "COM7"
        self.baud_rate = 9600
        self.connect_serial()
        time.sleep(2)
        self.state = False

    def connect_serial(self):
        if self.serial:
            if self.com_port is None:
                print("Serial port init error: COM port is unset.")
                return
            try:
                import serial
                self.connection = serial.Serial(self.com_port, baudrate=self.baud_rate, timeout=0,
                                                parity=serial.PARITY_NONE,
                                                stopbits=serial.STOPBITS_ONE,
                                                bytesize=serial.EIGHTBITS)
                print("Connection to port {} opened.".format(self.com_port))
            except:
                print("Serial port init error: Connecting to port {} failed.".format(self.com_port))

    def serial_loop(self):
        if self.connection:
            self.connection.write(b"0")
            time.sleep(0.1)
            lines = self.connection.readlines()
            #self.connection.send_break(constants.PULSE_DURATION_IN_MILLISECONDS/1000)
            #print(lines)
            if lines and not self.state and lines != [b'\x00']:
                self.state = True
                self.send_pedal(self.state)
            elif self.state and (not lines or lines == [b'\x00']):
                self.state = False
                self.send_pedal(self.state)

    def _midi_to_pedal(self, msg):
        # TODO: At this stage, interpret all note_on messages as the pedal being pressed,
        #       and note_off messages as the pedal being released. Later, use the correct
        #       message types and be more stringent about the messages.
        #
        if msg.type == 'note_on':
            state = True

        elif msg.type == 'note_off':
            state = False

        else:
            print("Unknown message type received from MIDI device")
            return
        self.send_pedal(state)

    def send_pedal(self, state):
        for callback_info in self._callback_infos:
            callback = callback_info['callback']
            callback(state)

        if not state:
            self._callback_infos = [callback_info for callback_info in self._callback_infos if not callback_info['remove_when_released']]

    def _connect_if_disconnected(self):
        if self._midi_in is None and len(self._midi_inputs) > 0:
            self._active_input = self._midi_inputs[0]
            self._midi_in = mido.open_input(self._active_input)
            self._midi_in._rt.ignore_types(False, False, False)
            self._midi_in.callback = self._midi_to_pedal

            Publisher.sendMessage('Pedal connection', state=True)

            print("Connected to MIDI device")

    def _check_disconnected(self):
        if self._midi_in is not None:
            if self._active_input not in self._midi_inputs:
                self._midi_in = None

                Publisher.sendMessage('Pedal connection', state=False)

                print("Disconnected from MIDI device")

    def _update_midi_inputs(self):
        self._midi_inputs = mido.get_input_names()

    def is_connected(self):
        return self._midi_in is not None

    def add_callback(self, name, callback, remove_when_released=False):
        self._callback_infos.append({
            'name': name,
            'callback': callback,
            'remove_when_released': remove_when_released,
        })

    def remove_callback(self, name):
        self._callback_infos = [callback_info for callback_info in self._callback_infos if callback_info['name'] != name]

    def run(self):
        self.in_use = True
        while True:
            #self._update_midi_inputs()
            #self._check_disconnected()
            #self._connect_if_disconnected()
            self.serial_loop()

            time.sleep(0.2)
