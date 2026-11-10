' Starts the H1 runner with no window (0 = hidden). Stop it with stop_h1_runner.bat.
Set sh = CreateObject("WScript.Shell")
dir = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
sh.Run "cmd /c """ & dir & "\run_h1_runner_bg.bat""", 0, False
