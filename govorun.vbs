Set sh = CreateObject("WScript.Shell")
sh.Run "pythonw """ & CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName) & "\govorun_pc.py""", 0, False
