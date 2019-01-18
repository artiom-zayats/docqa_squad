import smtplib
import os
import pdb
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from typing import Optional
 


def send_email(text_mesage:str , title_text:str , image_path:Optional[str]=None):

	# Define these once; use them twice!
	strFrom = 'azayats.updates@gmail.com'
	strTo = 'artiom.handro@gmail.com'

	# Create the root message and fill in the from, to, and subject headers
	msgRoot = MIMEMultipart('related')
	msgRoot['Subject'] = title_text
	msgRoot['From'] = strFrom
	msgRoot['To'] = strTo
	msgRoot.preamble = 'This is a multi-part message in MIME format.'
 
	# Encapsulate the plain and HTML versions of the message body in an
	# 'alternative' part, so message agents can decide which they want to display.
	msgAlternative = MIMEMultipart('alternative')
	msgRoot.attach(msgAlternative)

	msgText = MIMEText(text_mesage,'html')
	msgAlternative.attach(msgText)

	if image_path is not None:
		# We reference the image in the IMG SRC attribute by the ID we give it below
		msgText = MIMEText(text_mesage+'<br><img src="cid:image1"><br>'+ 'Enjoy1', 'html')
		msgAlternative.attach(msgText)

		# This example assumes the image is in the current directory
		fp = open(image_path, 'rb')
		msgImage = MIMEImage(fp.read())
		fp.close()

		# Define the image's ID as referenced above
		msgImage.add_header('Content-ID', '<image1>')
		msgRoot.attach(msgImage)




	server = smtplib.SMTP('smtp.gmail.com', 587)
	server.starttls()
	server.login(strFrom, get_pass())
	server.sendmail(strFrom, strTo, msgRoot.as_string())
	server.quit()


	print('Email to artiom.handro@gmail.com sent\n')

def get_pass():
	password = None
	pass_dir = os.path.expanduser('~/azayats/azayats_updates_email')
	pass_file_name = 'email_pass.txt'
	pass_path = os.path.join(pass_dir,pass_file_name)

	with open(pass_path, 'r') as f:
		first_line = f.readline()

	password = first_line[:-1]

	return str(password)


def create_train_dev_plot(dev_acc,train_acc,image_path):

	line1, = plt.plot(dev_acc, label="dev acc", color='darkgreen',linewidth=2 ,)
	line2, = plt.plot(train_acc, label="train acc", linewidth=2 ,color='darkblue')
	plt.legend([line1, line2], ["dev acc" , "train acc"])

	plt.ylabel('accuracy')
	plt.xlabel('evals')
	plt.savefig(os.path.join(image_path,'acc_results.png'))
	return os.path.join(image_path,'acc_results.png')
