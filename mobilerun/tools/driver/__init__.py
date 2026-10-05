"""Device driver compat surface for Mobilerun.

All driver implementations live in ``mobilerun-core-local`` (cloud ones
behind its ``[cloud]`` extra). Framework code imports them from there
directly; this package is kept only so the historical public
``mobilerun.tools.driver`` import path keeps working for external consumers.

The one repo-local exception is ``mobilerun.tools.driver.android_world``:
the AndroidWorld docker-server driver lives here because that HTTP API is
project-specific, not part of the shared core package.
"""

from mobilerun_core_local.driver.android import AndroidDriver
from mobilerun_core_local.driver.base import DeviceDisconnectedError, DeviceDriver
from mobilerun_core_local.driver.cloud import CloudDriver
from mobilerun_core_local.driver.ios import IOSDriver
from mobilerun_core_local.driver.recording import RecordingDriver
from mobilerun_core_local.driver.stealth import StealthDriver
from mobilerun_core_local.driver.visual_remote import VisualRemoteDriver

__all__ = [
    "DeviceDisconnectedError",
    "DeviceDriver",
    "AndroidDriver",
    "CloudDriver",
    "IOSDriver",
    "RecordingDriver",
    "StealthDriver",
    "VisualRemoteDriver",
]
