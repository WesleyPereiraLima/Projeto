from google.colab import drive
drive.mount('/content/drive')

# ====================================================
# Configurações de diretório
# ====================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_DIR = '/content/drive/MyDrive/dataset/'

OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

OUTPUT_DIR1 = './up_by_elevator'
if not os.path.exists(OUTPUT_DIR1):
    os.makedirs(OUTPUT_DIR1)

OUTPUT_DIR2 = './down_by_elevator'
if not os.path.exists(OUTPUT_DIR2):
    os.makedirs(OUTPUT_DIR2)


from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(INPUT_DIR) if isfile(join(INPUT_DIR, f))]

ACTIVITY_UP_ELEVATOR = 'up_by_elevator'
ACTIVITY_DOWN_ELEVATOR = 'down_by_elevator'

from types import AsyncGeneratorType
from scipy.integrate import quad
import math

columns = ['gyroscope_right_foot_x', 'gyroscope_left_foot_x','gyroscope_right_foot_y', 'gyroscope_left_foot_y','gyroscope_right_foot_z', 'gyroscope_left_foot_z', 'gyroscope_right_shin_x','gyroscope_left_shin_x','gyroscope_right_shin_y','gyroscope_left_shin_y','gyroscope_right_shin_z','gyroscope_left_shin_z','gyroscope_right_thigh_x','gyroscope_left_thigh_x','gyroscope_right_thigh_y','gyroscope_left_thigh_y','gyroscope_right_thigh_z','gyroscope_left_thigh_z']
gx= ['gyroscope_right_foot_x', 'gyroscope_left_foot_x','gyroscope_right_shin_x','gyroscope_left_shin_x','gyroscope_right_thigh_x','gyroscope_left_thigh_x']
gy= ['gyroscope_right_foot_y', 'gyroscope_left_foot_y','gyroscope_right_shin_y','gyroscope_left_shin_y','gyroscope_right_thigh_y','gyroscope_left_thigh_y']
gz= ['gyroscope_right_foot_z', 'gyroscope_left_foot_z','gyroscope_right_shin_z','gyroscope_left_shin_z','gyroscope_right_thigh_z','gyroscope_left_thigh_z']
ax = ['accelerometer_right_foot_x', 'accelerometer_left_foot_x','accelerometer_right_shin_x', 'accelerometer_left_shin_x', 'accelerometer_right_thigh_x', 'accelerometer_left_thigh_x']
ay = ['accelerometer_right_foot_y', 'accelerometer_left_foot_y','accelerometer_right_shin_y','accelerometer_left_shin_y', 'accelerometer_right_thigh_y', 'accelerometer_left_thigh_y']
az = ['accelerometer_right_foot_z','accelerometer_left_foot_z', 'accelerometer_right_shin_z', 'accelerometer_left_shin_z', 'accelerometer_right_thigh_z', 'accelerometer_left_thigh_z']
fus = []
agx = []
agy = []
agz = []
indx = []
indy = []
indz = []
sumx = []
sumy = []
sumz = []
anormx = []
anormy = []
anormz = []



def calculate_angles_x(x,y,z):
    axq = math.atan(x / (math.sqrt(z**2) + math.sqrt(y**2)))
    return axq
def calculate_angles_y(x,y,z):
    ayq = math.atan(y / (math.sqrt(z**2) + math.sqrt(x**2)))
    return ayq
def calculate_angles_z(x, y, z):
    azq = math.atan(z / (math.sqrt(x**2) + math.sqrt(y**2)))
    return azq

def update_angles():
  for file in onlyfiles:
    df = pd.read_csv(INPUT_DIR + file)
    for index in range(0,len(ax)):
      coluna_x = ax[index]
      coluna_y = ay[index]
      coluna_z = az[index]
      for line in range(0,df.shape[0]):
        x = df[f'{coluna_x}'][line]
        y = df[f'{coluna_y}'][line]
        z = df[f'{coluna_z}'][line]
        result_x = calculate_angles_x(x,y,z)
        df.loc[line, coluna_x] = result_x

        result_y = calculate_angles_y(x,y,z)
        df.loc[line, coluna_y] = result_y

        result_z = calculate_angles_z(x,y,z)
        df.loc[line, coluna_z] = result_z
    datab = df
    break
  return datab

datab = update_angles()


def omega(t):
    return 0.134  # velocidade angular constante

def calculate_angle(omega, t):
    integrand = lambda t: math.radians(omega(t))
    result, _ = quad(integrand, 0, t)
    return math.degrees(result)
def angulos_euler_x(gx):
    xg= math.atan(math.sin(gx)/math.cos(gx))
def angulos_euler_y(gy,gz):
    yg= math.sin(gy)/(math.sqrt(math.pow((math.cos(gz)) * math.cos(gy),2)) + math.sqrt(math.pow((math.sin(gz)) * math.cos(gy),2)))
def angulos_euler_z(gz):
    zg= math.atan(math.sin(gz)/math.cos(gz))

for file in onlyfiles:
    df = datab
    for col in columns:
      for line in range(0,df.shape[0]):
         x = df[f'{col}'][line]
         result_x = calculate_angle(omega,0.2)
         result = x*result_x
         df.loc[line, col] = result
    break

def update_angles_euler():
    data = df
    for index in range(0,len(ax)):
      coluna_x = gx[index]
      coluna_y = gy[index]
      coluna_z = gz[index]
      for line in range(0,df.shape[0]):
        x = data[f'{coluna_x}'][line]
        y = data[f'{coluna_y}'][line]
        z = data[f'{coluna_z}'][line]
        result_x = angulos_euler_x(x)
        data.loc[line, coluna_x] = result_x

        result_y = angulos_euler_y(y,z)
        data.loc[line, coluna_y] = result_y

        result_z = angulos_euler_z(z)
        data.loc[line, coluna_z] = result_z
      return df
      break
data = update_angles_euler()


for index in range(len(ax)):
    coluna_x = ax[index]
    coluna_y = ay[index]
    coluna_z = az[index]
    coluna_x1 = gx[index]
    coluna_y1 = gy[index]
    coluna_z1 = gz[index]

    for line in range(df.shape[0]):
        x = data.loc[line, coluna_x]
        y = data.loc[line, coluna_y]
        z = data.loc[line, coluna_z]
        x1 = data.loc[line, coluna_x1]
        y1 = data.loc[line, coluna_y1]
        z1 = data.loc[line, coluna_z1]

        result_x = 0.8 * x1 + 0.2 * x
        data.loc[line, coluna_x] = result_x

        result_y = 0.8 * y1 + 0.2 * y
        data.loc[line, coluna_y] = result_y

        result_z = 0.8 * z1 + 0.2 * z
        data.loc[line, coluna_z] = result_z

        fus.append(result_x)
        fus.append(result_y)
        fus.append(result_z)
        indx.append(result_x)
        indy.append(result_y)
        indz.append(result_z)

for i in range(0, len(fus), 3):
    agx.append(fus[i])
    agy.append(fus[i + 1])
    agz.append(fus[i + 2])

print(agx)  # Resultados 'x'
print(agy)  # Resultados 'y'
print(agz)  # Resultados 'z'

sumx = 0
sumy = 0
sumz = 0

num_elementos = len(agx) - 2436
for l in range(2436, len(agx)):
  sumx += agx[l]
  sumy += agy[l]
  sumz += agz[l]
mediax = sumx / num_elementos
mediay = sumy / num_elementos
mediaz = sumz / num_elementos

for g in range(2436, len(agx)):
  if (agx[g] + 5) > 6 or (agx[g] - 5) < -6:
    anormx.append(agx[g])
  if (agy[g] + 10) > 11 or (agy[g] - 10) < -11:
    anormy.append(agy[g])
  if (agz[g] + 5) > 6 or (agz[g] - 5) < -6:
    anormz.append(agz[g])

print('A média dos resultados de X é', mediax)
print('A média dos resultados de Y é', mediay)
print('A média dos resultados de Z é', mediaz)
print(anormx)
print(anormy)
print(anormz)
print(len(anormx))

for k in range(0, len(indx),6):
    if indx[k] > 0 and (k + 1) < len(indx) and indx[k + 1] < 0 or indx[k] < 0 and (k + 1) < len(indx) and indx[k + 1] > 0:
        print(f'Na posição {k} está fazendo o movimento de Andar')
    if indy[k] > 0 and (k + 1) < len(indy) and indy[k + 1] < 0:
        print(f'Na posição {k} está fazendo o movimento de Abrir as pernas')
    if indy[k] < 0 and (k + 1) < len(indy) and indy[k + 1] > 0:
        print(f'Na posição {k} está fazendo o movimento de Fechar as pernas')


# cria as subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))

# plota o resultado de agx
ax1.scatter(range(len(agx)), agx)
ax1.set_ylabel('Resultado x')

# plota o resultado de agy
ax2.scatter(range(len(agy)), agy)
ax2.set_ylabel('Resultado y')

# plota o resultado de agz
ax3.scatter(range(len(agz)), agz)
ax3.set_ylabel('Resultado z')

# define o título da figura
fig.suptitle('Resultado dos dados')

# ajusta o espaçamento entre subplots
fig.tight_layout()

# exibe a figura
plt.show()
