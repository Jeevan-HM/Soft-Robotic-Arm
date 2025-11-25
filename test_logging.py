import unittest
from unittest.mock import MagicMock, patch, call
import sys
import os
import threading
import time
import numpy as np

# Add current directory to path
sys.path.append(os.getcwd())

# Mock h5py before importing main
mock_h5py = MagicMock()
sys.modules["h5py"] = mock_h5py

import main

class TestLogging(unittest.TestCase):
    def setUp(self):
        self.mock_logger_patcher = patch('main.logger')
        self.mock_logger = self.mock_logger_patcher.start()
        
        # Reset global state if needed
        main.USE_MOCAP = False
        main.MOCAP_AVAILABLE = False

    def tearDown(self):
        self.mock_logger_patcher.stop()

    def test_periodic_flush(self):
        """Verify data is flushed to HDF5 periodically"""
        logger = main.DataLogger()
        logger.hdf5_path = "test.h5"
        logger.exp_group_name = "exp_test"
        logger.flush_interval = 5
        logger.start_time = time.time()
        
        # Mock HDF5 file and dataset
        mock_file = MagicMock()
        mock_grp = MagicMock()
        mock_dset = MagicMock()
        mock_dset.shape = (0, 10)
        
        mock_h5py.File.return_value.__enter__.return_value = mock_file
        mock_file.__getitem__.return_value = mock_grp
        mock_grp.__getitem__.return_value = mock_dset
        
        # Log 5 samples (should trigger flush)
        for i in range(5):
            logger.log([0.0]*4, [[0.0]*4]*4)
            
        # Verify resize and write called
        mock_dset.resize.assert_called()
        self.assertEqual(len(logger.data_buffer), 0) # Buffer should be empty

    def test_thread_locking(self):
        """Verify lock is acquired during critical sections"""
        controller = main.Controller()
        controller.arduino = MagicMock()
        controller.arduino.send_pressure.return_value = [0.0]*4
        
        # Mock the lock
        controller.data_lock = MagicMock()
        
        # Test send_all acquires lock
        controller.send_all()
        controller.data_lock.__enter__.assert_called()
        
        # Test wave loop acquires lock (simulate one iteration)
        # We can't easily test the loop itself, but we can check if modifying desired uses lock
        # Let's manually call the logic inside a wave function
        with patch('main.time.sleep'): # Don't sleep
            main.axial(controller)
            # Verify lock was used multiple times
            self.assertTrue(controller.data_lock.__enter__.call_count > 0)

    def test_nan_on_error(self):
        """Verify ArduinoManager returns NaNs on error"""
        manager = main.ArduinoManager()
        mock_client = MagicMock()
        mock_client.send.side_effect = Exception("Connection lost")
        manager.client_sockets = [mock_client] * 4
        
        sensors = manager.send_pressure(0, 10.0)
        
        self.assertTrue(np.isnan(sensors).all())
        self.assertEqual(len(sensors), 4)

if __name__ == '__main__':
    unittest.main()
