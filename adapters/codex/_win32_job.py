"""Narrow Win32 Job Object launcher for controlled Codex native Exec.

Every launched root is assigned to one unnamed Job at process creation through
``PROC_THREAD_ATTRIBUTE_JOB_LIST``.  The returned object owns only that Job,
the root process handle, and the three parent pipe endpoints.
"""
from __future__ import annotations

import asyncio
import contextlib
import ctypes
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import BinaryIO

if os.name != "nt":

    class JobProcess:
        """Import-only type surface; native launch is unavailable off Windows."""

    def create_process_in_job(
        argv: Sequence[str],
        *,
        cwd: str | Path,
        env: Mapping[str, str],
    ) -> JobProcess:
        raise OSError("codex-native-exec requires Windows Job Objects")

else:
    import msvcrt
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    _HANDLE = wintypes.HANDLE
    _DWORD = wintypes.DWORD
    _BOOL = wintypes.BOOL
    _LPVOID = wintypes.LPVOID
    _SIZE_T = ctypes.c_size_t
    _ULONG_PTR = ctypes.c_size_t

    _CREATE_SUSPENDED = 0x00000004
    _CREATE_UNICODE_ENVIRONMENT = 0x00000400
    _EXTENDED_STARTUPINFO_PRESENT = 0x00080000
    _STARTF_USESTDHANDLES = 0x00000100
    _PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x00020002
    _PROC_THREAD_ATTRIBUTE_JOB_LIST = 0x0002000D
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x00000800
    _JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK = 0x00001000
    _JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1
    _JOB_OBJECT_BASIC_PROCESS_ID_LIST = 3
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    _HANDLE_FLAG_INHERIT = 0x00000001
    _WAIT_OBJECT_0 = 0x00000000
    _WAIT_TIMEOUT = 0x00000102
    _WAIT_FAILED = 0xFFFFFFFF
    _INFINITE = 0xFFFFFFFF
    _STILL_ACTIVE = 259

    class _SecurityAttributes(ctypes.Structure):
        _fields_ = [
            ("nLength", _DWORD),
            ("lpSecurityDescriptor", _LPVOID),
            ("bInheritHandle", _BOOL),
        ]

    class _StartupInfoW(ctypes.Structure):
        _fields_ = [
            ("cb", _DWORD),
            ("lpReserved", wintypes.LPWSTR),
            ("lpDesktop", wintypes.LPWSTR),
            ("lpTitle", wintypes.LPWSTR),
            ("dwX", _DWORD),
            ("dwY", _DWORD),
            ("dwXSize", _DWORD),
            ("dwYSize", _DWORD),
            ("dwXCountChars", _DWORD),
            ("dwYCountChars", _DWORD),
            ("dwFillAttribute", _DWORD),
            ("dwFlags", _DWORD),
            ("wShowWindow", wintypes.WORD),
            ("cbReserved2", wintypes.WORD),
            ("lpReserved2", ctypes.POINTER(wintypes.BYTE)),
            ("hStdInput", _HANDLE),
            ("hStdOutput", _HANDLE),
            ("hStdError", _HANDLE),
        ]

    class _StartupInfoExW(ctypes.Structure):
        _fields_ = [("StartupInfo", _StartupInfoW), ("lpAttributeList", _LPVOID)]

    class _ProcessInformation(ctypes.Structure):
        _fields_ = [
            ("hProcess", _HANDLE),
            ("hThread", _HANDLE),
            ("dwProcessId", _DWORD),
            ("dwThreadId", _DWORD),
        ]

    class _BasicAccountingInformation(ctypes.Structure):
        _fields_ = [
            ("TotalUserTime", ctypes.c_longlong),
            ("TotalKernelTime", ctypes.c_longlong),
            ("ThisPeriodTotalUserTime", ctypes.c_longlong),
            ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
            ("TotalPageFaultCount", _DWORD),
            ("TotalProcesses", _DWORD),
            ("ActiveProcesses", _DWORD),
            ("TotalTerminatedProcesses", _DWORD),
        ]

    class _BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", _DWORD),
            ("MinimumWorkingSetSize", _SIZE_T),
            ("MaximumWorkingSetSize", _SIZE_T),
            ("ActiveProcessLimit", _DWORD),
            ("Affinity", _ULONG_PTR),
            ("PriorityClass", _DWORD),
            ("SchedulingClass", _DWORD),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", _SIZE_T),
            ("JobMemoryLimit", _SIZE_T),
            ("PeakProcessMemoryUsed", _SIZE_T),
            ("PeakJobMemoryUsed", _SIZE_T),
        ]

    _kernel32.CreatePipe.argtypes = [
        ctypes.POINTER(_HANDLE),
        ctypes.POINTER(_HANDLE),
        ctypes.POINTER(_SecurityAttributes),
        _DWORD,
    ]
    _kernel32.CreatePipe.restype = _BOOL
    _kernel32.SetHandleInformation.argtypes = [_HANDLE, _DWORD, _DWORD]
    _kernel32.SetHandleInformation.restype = _BOOL
    _kernel32.CreateJobObjectW.argtypes = [_LPVOID, wintypes.LPCWSTR]
    _kernel32.CreateJobObjectW.restype = _HANDLE
    _kernel32.SetInformationJobObject.argtypes = [_HANDLE, ctypes.c_int, _LPVOID, _DWORD]
    _kernel32.SetInformationJobObject.restype = _BOOL
    _kernel32.QueryInformationJobObject.argtypes = [
        _HANDLE,
        ctypes.c_int,
        _LPVOID,
        _DWORD,
        ctypes.POINTER(_DWORD),
    ]
    _kernel32.QueryInformationJobObject.restype = _BOOL
    _kernel32.InitializeProcThreadAttributeList.argtypes = [
        _LPVOID,
        _DWORD,
        _DWORD,
        ctypes.POINTER(_SIZE_T),
    ]
    _kernel32.InitializeProcThreadAttributeList.restype = _BOOL
    _kernel32.UpdateProcThreadAttribute.argtypes = [
        _LPVOID,
        _DWORD,
        _SIZE_T,
        _LPVOID,
        _SIZE_T,
        _LPVOID,
        ctypes.POINTER(_SIZE_T),
    ]
    _kernel32.UpdateProcThreadAttribute.restype = _BOOL
    _kernel32.DeleteProcThreadAttributeList.argtypes = [_LPVOID]
    _kernel32.DeleteProcThreadAttributeList.restype = None
    _kernel32.CreateProcessW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPWSTR,
        _LPVOID,
        _LPVOID,
        _BOOL,
        _DWORD,
        _LPVOID,
        wintypes.LPCWSTR,
        ctypes.POINTER(_StartupInfoW),
        ctypes.POINTER(_ProcessInformation),
    ]
    _kernel32.CreateProcessW.restype = _BOOL
    _kernel32.IsProcessInJob.argtypes = [_HANDLE, _HANDLE, ctypes.POINTER(_BOOL)]
    _kernel32.IsProcessInJob.restype = _BOOL
    _kernel32.WaitForSingleObject.argtypes = [_HANDLE, _DWORD]
    _kernel32.WaitForSingleObject.restype = _DWORD
    _kernel32.GetExitCodeProcess.argtypes = [_HANDLE, ctypes.POINTER(_DWORD)]
    _kernel32.GetExitCodeProcess.restype = _BOOL
    _kernel32.ResumeThread.argtypes = [_HANDLE]
    _kernel32.ResumeThread.restype = _DWORD
    _kernel32.TerminateJobObject.argtypes = [_HANDLE, wintypes.UINT]
    _kernel32.TerminateJobObject.restype = _BOOL
    _kernel32.CloseHandle.argtypes = [_HANDLE]
    _kernel32.CloseHandle.restype = _BOOL

    def _winerror(api: str) -> OSError:
        return ctypes.WinError(ctypes.get_last_error(), api)

    def _require(ok: object, api: str) -> None:
        if not ok:
            raise _winerror(api)

    def _close_handle(handle: int | None) -> None:
        if handle:
            _kernel32.CloseHandle(handle)

    def _wait_handle(handle: int, timeout_ms: int) -> int:
        result = int(_kernel32.WaitForSingleObject(handle, timeout_ms))
        if result == _WAIT_FAILED:
            raise _winerror("WaitForSingleObject")
        return result

    def _query_active_processes(job: int) -> int:
        info = _BasicAccountingInformation()
        returned = _DWORD()
        _require(
            _kernel32.QueryInformationJobObject(
                job,
                _JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
                ctypes.byref(info),
                ctypes.sizeof(info),
                ctypes.byref(returned),
            ),
            "QueryInformationJobObject(Accounting)",
        )
        return int(info.ActiveProcesses)

    def _query_process_ids(job: int) -> tuple[int, ...]:
        capacity = 4096
        while capacity <= 1024 * 1024:
            buffer = ctypes.create_string_buffer(capacity)
            returned = _DWORD()
            if _kernel32.QueryInformationJobObject(
                job,
                _JOB_OBJECT_BASIC_PROCESS_ID_LIST,
                buffer,
                capacity,
                ctypes.byref(returned),
            ):
                listed = _DWORD.from_buffer_copy(buffer.raw[4:8]).value
                array_type = _ULONG_PTR * listed
                if not listed:
                    return ()
                end = 8 + listed * ctypes.sizeof(_ULONG_PTR)
                return tuple(int(pid) for pid in array_type.from_buffer_copy(buffer.raw[8:end]))
            if ctypes.get_last_error() != 234:
                raise _winerror("QueryInformationJobObject(ProcessIdList)")
            capacity *= 2
        raise RuntimeError("Job process-ID list exceeded bounded 1 MiB buffer")

    def _process_in_job(process: int, job: int) -> bool:
        answer = _BOOL()
        _require(
            _kernel32.IsProcessInJob(process, job, ctypes.byref(answer)),
            "IsProcessInJob",
        )
        return bool(answer.value)

    def _create_pipe() -> tuple[int, int]:
        read_handle = _HANDLE()
        write_handle = _HANDLE()
        attributes = _SecurityAttributes(
            nLength=ctypes.sizeof(_SecurityAttributes),
            lpSecurityDescriptor=None,
            bInheritHandle=True,
        )
        _require(
            _kernel32.CreatePipe(
                ctypes.byref(read_handle),
                ctypes.byref(write_handle),
                ctypes.byref(attributes),
                0,
            ),
            "CreatePipe",
        )
        return int(read_handle.value), int(write_handle.value)

    def _make_parent_file(handle: int, flags: int, mode: str) -> BinaryIO:
        descriptor = msvcrt.open_osfhandle(handle, flags | os.O_BINARY)
        try:
            return os.fdopen(descriptor, mode, buffering=0)
        except BaseException:
            os.close(descriptor)
            raise

    def _environment_block(env: Mapping[str, str]) -> ctypes.Array[ctypes.c_wchar]:
        entries: list[str] = []
        for key, value in env.items():
            if "\0" in key or "\0" in value:
                raise ValueError("environment contains NUL")
            entries.append(f"{key}={value}")
        entries.sort(key=str.casefold)
        return ctypes.create_unicode_buffer("\0".join(entries) + "\0\0")

    class _AsyncPipeReader:
        def __init__(self, stream: BinaryIO) -> None:
            self._stream = stream

        async def read(self, size: int = -1) -> bytes:
            return await asyncio.to_thread(self._stream.read, size)

        async def readline(self, size: int = -1) -> bytes:
            return await asyncio.to_thread(self._stream.readline, size)

        def close(self) -> None:
            self._stream.close()

    class _AsyncPipeWriter:
        def __init__(self, stream: BinaryIO) -> None:
            self._stream = stream
            self._pending = bytearray()
            self._closed = False

        def write(self, data: bytes) -> None:
            if self._closed:
                raise ValueError("write to closed process stdin")
            self._pending.extend(data)

        async def drain(self) -> None:
            if self._closed:
                raise ValueError("drain of closed process stdin")
            if not self._pending:
                return
            data = bytes(self._pending)
            await asyncio.to_thread(self._write_all, data)
            del self._pending[: len(data)]

        def _write_all(self, data: bytes) -> None:
            written = 0
            view = memoryview(data)
            while written < len(view):
                count = self._stream.write(view[written:])
                if count is None or count <= 0:
                    raise BrokenPipeError("process stdin write made no progress")
                written += count
            self._stream.flush()

        def close(self) -> None:
            if not self._closed:
                self._closed = True
                self._stream.close()

    class JobProcess:
        """Exact ownership bundle for one root process and its unnamed Job."""

        def __init__(
            self,
            *,
            job_handle: int,
            process_handle: int,
            pid: int,
            stdin: BinaryIO,
            stdout: BinaryIO,
            stderr: BinaryIO,
            pre_resume_process_ids: tuple[int, ...],
        ) -> None:
            self._job_handle = job_handle
            self._process_handle = process_handle
            self.pid = pid
            self.stdin = _AsyncPipeWriter(stdin)
            self.stdout = _AsyncPipeReader(stdout)
            self.stderr = _AsyncPipeReader(stderr)
            self.pre_resume_process_ids = pre_resume_process_ids
            self.creation_time_membership_verified = True
            self.returncode: int | None = None
            self._closed = False
            self._wait_lock = asyncio.Lock()
            self.terminate_job_calls = 0
            self.active_processes_at_close: int | None = None
            self.kill_on_job_close_enabled = True
            self.breakaway_flags_disabled = True

        @property
        def closed(self) -> bool:
            return self._closed

        def active_processes(self) -> int:
            if self._closed:
                raise RuntimeError("Job handle is closed")
            return _query_active_processes(self._job_handle)

        def process_ids(self) -> tuple[int, ...]:
            if self._closed:
                raise RuntimeError("Job handle is closed")
            return _query_process_ids(self._job_handle)

        async def wait(self) -> int:
            async with self._wait_lock:
                if self.returncode is not None:
                    return self.returncode
                result = await asyncio.to_thread(_wait_handle, self._process_handle, _INFINITE)
                if result != _WAIT_OBJECT_0:
                    raise RuntimeError(f"unexpected process wait result {result}")
                code = _DWORD()
                _require(
                    _kernel32.GetExitCodeProcess(self._process_handle, ctypes.byref(code)),
                    "GetExitCodeProcess",
                )
                if code.value == _STILL_ACTIVE:
                    raise RuntimeError("process signalled but still reports active")
                self.returncode = int(code.value)
                return self.returncode

        def terminate_job(self, exit_code: int = 1) -> None:
            if not self._closed:
                self.terminate_job_calls += 1
                _require(
                    _kernel32.TerminateJobObject(self._job_handle, exit_code),
                    "TerminateJobObject",
                )

        def close(self) -> None:
            if self._closed:
                return
            errors: list[BaseException] = []
            try:
                self.active_processes_at_close = _query_active_processes(self._job_handle)
            except BaseException as exc:
                errors.append(exc)
            self._closed = True
            for stream in (self.stdin, self.stdout, self.stderr):
                try:
                    stream.close()
                except BaseException as exc:
                    errors.append(exc)
            for handle in (self._process_handle, self._job_handle):
                if handle and not _kernel32.CloseHandle(handle):
                    errors.append(_winerror("CloseHandle"))
            self._process_handle = 0
            self._job_handle = 0
            if errors:
                raise errors[0]

        def __del__(self) -> None:
            if getattr(self, "_closed", True):
                return
            with contextlib.suppress(BaseException):
                _kernel32.TerminateJobObject(self._job_handle, 1)
            with contextlib.suppress(BaseException):
                self.close()

    def create_process_in_job(
        argv: Sequence[str],
        *,
        cwd: str | Path,
        env: Mapping[str, str],
    ) -> JobProcess:
        """Create a suspended root already assigned to one exact unnamed Job."""
        if not argv or not all(isinstance(part, str) and part for part in argv):
            raise ValueError("argv must contain non-empty strings")

        job: int | None = None
        process_handle: int | None = None
        thread_handle: int | None = None
        attribute_buffer: ctypes.Array[ctypes.c_char] | None = None
        attributes_initialized = False
        pipe_handles: set[int] = set()
        parent_streams: list[BinaryIO] = []
        resumed = False
        try:
            job_value = _kernel32.CreateJobObjectW(None, None)
            _require(job_value, "CreateJobObjectW")
            job = int(job_value)

            limits = _ExtendedLimitInformation()
            limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            _require(
                _kernel32.SetInformationJobObject(
                    job,
                    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                    ctypes.byref(limits),
                    ctypes.sizeof(limits),
                ),
                "SetInformationJobObject",
            )
            observed_limits = _ExtendedLimitInformation()
            returned = _DWORD()
            _require(
                _kernel32.QueryInformationJobObject(
                    job,
                    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                    ctypes.byref(observed_limits),
                    ctypes.sizeof(observed_limits),
                    ctypes.byref(returned),
                ),
                "QueryInformationJobObject(ExtendedLimit)",
            )
            flags = int(observed_limits.BasicLimitInformation.LimitFlags)
            forbidden = _JOB_OBJECT_LIMIT_BREAKAWAY_OK | _JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK
            if not flags & _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE or flags & forbidden:
                raise RuntimeError("Job limit configuration did not fail closed")

            stdin_read, stdin_write = _create_pipe()
            stdout_read, stdout_write = _create_pipe()
            stderr_read, stderr_write = _create_pipe()
            pipe_handles.update(
                {stdin_read, stdin_write, stdout_read, stdout_write, stderr_read, stderr_write}
            )
            for parent_handle in (stdin_write, stdout_read, stderr_read):
                _require(
                    _kernel32.SetHandleInformation(parent_handle, _HANDLE_FLAG_INHERIT, 0),
                    "SetHandleInformation",
                )

            attribute_size = _SIZE_T()
            ctypes.set_last_error(0)
            first = _kernel32.InitializeProcThreadAttributeList(
                None, 2, 0, ctypes.byref(attribute_size)
            )
            if first or ctypes.get_last_error() != 122:
                raise _winerror("InitializeProcThreadAttributeList(size query)")
            attribute_buffer = ctypes.create_string_buffer(attribute_size.value)
            _require(
                _kernel32.InitializeProcThreadAttributeList(
                    attribute_buffer, 2, 0, ctypes.byref(attribute_size)
                ),
                "InitializeProcThreadAttributeList",
            )
            attributes_initialized = True

            job_array = (_HANDLE * 1)(job)
            _require(
                _kernel32.UpdateProcThreadAttribute(
                    attribute_buffer,
                    0,
                    _PROC_THREAD_ATTRIBUTE_JOB_LIST,
                    ctypes.cast(job_array, _LPVOID),
                    ctypes.sizeof(job_array),
                    None,
                    None,
                ),
                "UpdateProcThreadAttribute(PROC_THREAD_ATTRIBUTE_JOB_LIST)",
            )
            inherited_handles = (_HANDLE * 3)(stdin_read, stdout_write, stderr_write)
            _require(
                _kernel32.UpdateProcThreadAttribute(
                    attribute_buffer,
                    0,
                    _PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                    ctypes.cast(inherited_handles, _LPVOID),
                    ctypes.sizeof(inherited_handles),
                    None,
                    None,
                ),
                "UpdateProcThreadAttribute(PROC_THREAD_ATTRIBUTE_HANDLE_LIST)",
            )

            startup = _StartupInfoExW()
            startup.StartupInfo.cb = ctypes.sizeof(startup)
            startup.StartupInfo.dwFlags = _STARTF_USESTDHANDLES
            startup.StartupInfo.hStdInput = stdin_read
            startup.StartupInfo.hStdOutput = stdout_write
            startup.StartupInfo.hStdError = stderr_write
            startup.lpAttributeList = ctypes.cast(attribute_buffer, _LPVOID)
            process_info = _ProcessInformation()
            command_line = ctypes.create_unicode_buffer(subprocess.list2cmdline(list(argv)))
            environment = _environment_block(env)
            _require(
                _kernel32.CreateProcessW(
                    None,
                    command_line,
                    None,
                    None,
                    True,
                    _CREATE_SUSPENDED
                    | _CREATE_UNICODE_ENVIRONMENT
                    | _EXTENDED_STARTUPINFO_PRESENT,
                    environment,
                    str(Path(cwd)),
                    ctypes.byref(startup.StartupInfo),
                    ctypes.byref(process_info),
                ),
                "CreateProcessW(STARTUPINFOEX + JOB_LIST)",
            )
            process_handle = int(process_info.hProcess)
            thread_handle = int(process_info.hThread)
            pid = int(process_info.dwProcessId)

            pre_resume_ids = _query_process_ids(job)
            if (
                not _process_in_job(process_handle, job)
                or _query_active_processes(job) != 1
                or pre_resume_ids != (pid,)
            ):
                raise RuntimeError("root Job membership was not proven before ResumeThread")

            if _kernel32.ResumeThread(thread_handle) == 0xFFFFFFFF:
                raise _winerror("ResumeThread")
            resumed = True
            _close_handle(thread_handle)
            thread_handle = None

            for child_handle in (stdin_read, stdout_write, stderr_write):
                _close_handle(child_handle)
                pipe_handles.discard(child_handle)

            stdin_stream = _make_parent_file(stdin_write, os.O_WRONLY, "wb")
            pipe_handles.discard(stdin_write)
            parent_streams.append(stdin_stream)
            stdout_stream = _make_parent_file(stdout_read, os.O_RDONLY, "rb")
            pipe_handles.discard(stdout_read)
            parent_streams.append(stdout_stream)
            stderr_stream = _make_parent_file(stderr_read, os.O_RDONLY, "rb")
            pipe_handles.discard(stderr_read)
            parent_streams.append(stderr_stream)

            owned = JobProcess(
                job_handle=job,
                process_handle=process_handle,
                pid=pid,
                stdin=stdin_stream,
                stdout=stdout_stream,
                stderr=stderr_stream,
                pre_resume_process_ids=pre_resume_ids,
            )
            job = None
            process_handle = None
            parent_streams.clear()
            return owned
        except BaseException:
            if job and (resumed or process_handle):
                _kernel32.TerminateJobObject(job, 1)
            for stream in parent_streams:
                with contextlib.suppress(OSError):
                    stream.close()
            raise
        finally:
            if attributes_initialized and attribute_buffer is not None:
                _kernel32.DeleteProcThreadAttributeList(attribute_buffer)
            _close_handle(thread_handle)
            for handle in pipe_handles:
                _close_handle(handle)
            _close_handle(process_handle)
            _close_handle(job)
