import time
from datetime import datetime

def alarm():
    print(datetime.now())

    alarm_time_str = input('Enter alarm time in format HH:MM:SS (24-hour): ')
    
    try:
        alarm_time = datetime.strptime(alarm_time_str, '%H:%M:%S').time()
    except ValueError:
        print("Invalid time format. Please use HH:MM:SS format.")
        return
    
    print(f'Alarm set for {alarm_time}')
    
    while True:
        current_time = datetime.now().time()  
        if current_time >= alarm_time:
            print('Beep! Alarm ringing...')
            break
        time.sleep(1)  

alarm()
