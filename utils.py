import streamlit as st
import datetime
def calculate_len_texts(text):
    words=text.split()
    return len(words)


def ciudad_a_poblacion(ciudad, ciudades):
    return ciudades[ciudades["DPMP"]==ciudad]["Población"].values[0]

def raie_if_not_colombia(pais):
    if pais != "Colombia":
        #print in streamlit that model only works in Colombia en rojo
        st.write("El modelo solo funciona en Colombia")
    else:
        pass
    return pais

def get_dat_variables(datetime_value):
    return datetime_value.weekday(),  datetime_value.day
    
def translate_date(date):
    """get weekday and day of the month"""
    return date.weekday(), date.day
def translate_time(time):
    if time > datetime.datetime.strptime("6:00:00", "%H:%M:%S").time() and time < datetime.datetime.strptime("12:00:00", "%H:%M:%S").time():
        return "morning"
    elif time >= datetime.datetime.strptime("12:00:00", "%H:%M:%S").time() and time < datetime.datetime.strptime("18:00:00", "%H:%M:%S").time():
        return "afternoon"
    else:
        return "night"
