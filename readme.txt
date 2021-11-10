Instructions for installations and use.
Download the repository:
Option 1: use the download link available in the site.

Option 2: use git clone “link”
Create and activate virtual environment 
First it’s necessary to create a virtual environment and activate it. Use the next command line:

Python3 -m venv “Virtualenv name”

Activate virtual environment

Source path of virtualenv”/bin/activate

Install the requirements packages:

Change at the folder where you download the files.

cd /path of files/

To install the necessary packages run the next command line:

pip install -r requirements.txt

Run the program.
Use the next command line:

python guiv1.1.py

If everything works fine, it will show the GUI without problem.

Issues solution

In case of this error shows:

“ qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "/home/pc/envs/”Virtualenv name”/lib/python3.8/site-packages/cv2/qt/plugins" even though it was found. “

Change to directory: envs/gui_speos/lib/python3.8/site-packages/cv2/qt/

Use the next command line: “cd envs/gui_speos/lib/python3.8/site-packages/cv2/qt/”

And delete the folder "plugins” 

For delete the folder plugins use the next command line:

“rm -rf plugins/”