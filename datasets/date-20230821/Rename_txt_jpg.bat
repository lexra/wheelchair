@echo off
setlocal enabledelayedexpansion

set "folder_path=C:\Users\raymond.kao\Desktop\himax\image\People_wheelchair_image\Wheel_chair_dataset\wheelchair_0821\train"

set "count=0"
for %%F in ("%folder_path%\*.jpg") do (
    set "filename=%%~nF"
    set "ext=%%~xF"
    set "txt_file=!folder_path!\!filename!.txt"
    if exist "!txt_file!" (
        set "formatted_count=000!count!"
        set "formatted_count=!formatted_count:~-5!"
        ren "%%F" "!formatted_count!!ext!"
        ren "!txt_file!" "!formatted_count!.txt"
        set /a "count+=1"
    )
)

echo Done!
