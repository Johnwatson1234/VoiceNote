; VoiceNote Windows 安装器
;
; 两个刻意的选择：
;
;   1. PrivilegesRequired=lowest + 装到 {localappdata}\Programs
;      → 全程不弹 UAC、不需要管理员。"一键安装"如果第一步就是提权对话框，
;        那就不叫一键了。副作用是每个用户各装一份，对本应用无所谓。
;
;   2. Compression=lzma2/ultra64
;      → 2.3GB 的产物能压到 1GB 出头。这不是为了好看：GitHub Release
;        单文件上限是 2GiB，压不下来就直接发不出去。
;
; AppVersion 由 scripts\build.ps1 通过 /DAppVersion= 传入。

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

#define AppName "VoiceNote"
#define AppPublisher "The VoiceNote Authors"
#define AppURL "https://github.com/Johnwatson1234/VoiceNote"
#define SourceDir "..\dist\VoiceNote"

[Setup]
AppId={{8F3A2C51-6D44-4E2B-9C1A-7B5E0D3F9A21}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=VoiceNote-Setup-{#AppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\VoiceNote.exe
; 装完目录里没有需要手工确认的选项，去掉目录页让流程更短
DisableDirPage=auto
DisableReadyPage=no

[Languages]
; 中文放前面，作为默认语言
Name: "chinese"; MessagesFile: "ChineseSimplified.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "autostart"; Description: "开机自动启动"; GroupDescription: "附加任务:"
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加任务:"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\VoiceNote.exe"
Name: "{group}\卸载 {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\VoiceNote.exe"; Tasks: desktopicon

[Registry]
; 和托盘菜单里的「开机自启」写的是同一个键值，两边不会打架
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; \
    ValueType: string; ValueName: "{#AppName}"; ValueData: """{app}\VoiceNote.exe"""; \
    Flags: uninsdeletevalue; Tasks: autostart

[Run]
Filename: "{app}\VoiceNote.exe"; Description: "立即启动 {#AppName}"; Flags: nowait postinstall skipifsilent
