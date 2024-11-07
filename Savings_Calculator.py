Saving_Per_Year = int(input("How much money do you want to save in a year? "))
Saving_Per_Month = (Saving_Per_Year / 12)
Saving_Per_Week = (Saving_Per_Year / 52)
Saving_Per_Day = (Saving_Per_Year / 365)
print("Based on how much you want to save per year you must save: \n"
      + str(round(Saving_Per_Month, 2)) + " per month, \n"
      + str(round(Saving_Per_Week, 2)) + " per week, and \n"
      + str(round(Saving_Per_Day, 2)) + " per day.")
