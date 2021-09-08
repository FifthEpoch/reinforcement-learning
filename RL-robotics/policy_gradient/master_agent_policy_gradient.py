import numpy as np
import serial
import math

# servant arduino
serial = serial.Serial('/dev/ttyACM0', 115200, timeout=0.5)

def byte_data_to_list():
    byte_data = serial.readline()
    data = byte_data[0:len(byte_data) - 2].decode("utf-8")
    data_list = np.array(data.split(), dtype=float)
    return data_list

file_path_dict = {
        0: 'w.txt',
        1: 'zw.txt',
        2: 'zt.txt',
        3: 'theta.txt',
        4: 'r.txt',
        5: 'tde.txt'
    }

def write_to_record(_mode, _avg_val):
    f_path = file_path_dict.get(_mode, 'r.txt')
    with open(f_path, "a") as record:
        record.write("\n{}".format(_avg_val))
    record.close()

def write_to_memory(_mode, _ndarray):
    f_path = file_path_dict.get(_mode, 'w.txt')
    with open(f_path, "w+") as memory:
        memory.truncate(0)
        row = ''
        for i in range(4):
            ndarr_i = np.ndarray.copy(_ndarray[i])
            arr_i = np.ndarray.flatten(ndarr_i)
            for j in range(arr_i.size):
                row += (str(arr_i[j]) + ' ')
            row += '\n'
        memory.write(row)
    memory.close()

def read_from_memory(_mode, _ndarray):
    f_path = file_path_dict.get(_mode, 'w.txt')
    with open(f_path, "r") as memory:
        data_list = np.array([])
        for i in range(4):
                data_row = np.array(memory.readline().split(), dtype=float)
                data_list = np.append(data_list, data_row)
        updated_ndarray = np.reshape(data_list, _ndarray.shape)
        return updated_ndarray

def bound_temp_data(_data):
    _data = 29.9 if _data >= 29.9 else _data
    _data = 23.0 if _data < 23.0 else _data
    return _data

def bound_photo_data(_data):
    _data = 511.0 if _data >= 511.0 else _data
    _data = 0.0 if _data < 0.0 else _data
    return _data

def bound_uss_data(_data):
    _data = 101.5 if _data >= 101.5 else _data
    _data = 2.0 if _data < 2.0 else _data
    return _data

def process_sensor_data(_data_list):

    l_photo = _data_list[0]
    r_photo = _data_list[1]
    l_photo = bound_photo_data(l_photo)
    r_photo = bound_photo_data(r_photo)

    temp = _data_list[2]
    temp = bound_temp_data(temp)

    watch30 = _data_list[3]
    watch135 = _data_list[4]
    watch90 = _data_list[5]
    uss = bound_uss_data((watch90 * 0.5) + (watch30 * 0.25) + (watch135 * 0.25))

    return l_photo, r_photo, temp, uss

def get_reward(_temp, _l_bump, _r_bump):
    r = 1.0 if (_temp >= 25.7 and _temp <= 26.2) else -1.0
    r = r - 5.0 if _l_bump else r
    r = r - 5.0 if _r_bump else r
    return r

if __name__ == '__main__':

    # Actor-Critic with Eligibility Traces (continuing)
    # for estimating pi_theta approx. equals to pi*
    # p.333 in Reinforcement Learning: An Introduction

    t = 0

    eps = 0.001

    lambda_w = 0.75
    lambda_t = 0.75

    alpha_w = 0.0025
    alpha_t = 0.0025
    alpha_r = 0.0025

    r_bar = 0.0

    feature_num = 8
    action_num = 5
    tile_num = 4

    # units per metric: width / tile_num
    photo_unit = 12.7875
    temp_unit = 0.5
    uss_unit = 5.0

    # w = [tilings][left photo][right photo][photo roc][temp][temp roc][sonic 90][l_bumper][r_bumper][action-num]
    #         4         10           10          2       10       2         5         2        2          5
    w = np.zeros((4, 15, 15, 2, 15, 2, 5, 2, 2, 5))
    # or load from memory
    # w = read_from_memory(True, w)

    theta = np.zeros((4, 15, 15, 2, 15, 2, 5, 2, 2, 5))
    # or load from memory
    # theta = read_from_memory(True, theta)

    z_w = np.zeros((4, 15, 15, 2, 15, 2, 5, 2, 2, 5))
    # or load from memory
    # z_w = read_from_memory(False, z_w)

    z_t = np.zeros((4, 15, 15, 2, 15, 2, 5, 2, 2, 5))
    # or load from memory
    # z_t = read_from_memory(False, z_t)

    x = np.zeros(feature_num, dtype=int)
    xp = np.zeros(feature_num, dtype=int)
    gen_x_i = np.zeros((tile_num, feature_num), dtype=int)
    gen_xp_i = np.zeros((tile_num, feature_num), dtype=int)

    a, ap = 4, 0
    q, qp = 0.0, 0.0
    td_err = 0.0

    avg_r = 0.0
    avg_td_err = 0.0

    last_photo = 0.0
    last_temp = 0.0

    # init. last_photo and last_temp

    while serial.in_waiting <= 5:
        continue

    data_list = byte_data_to_list()

    l_photo, r_photo, temp, uss = process_sensor_data(data_list)
    avg_photo = (l_photo + r_photo) / 2.0

    l_bump = int(data_list[6])
    r_bump = int(data_list[7])

    x[0] = math.floor((l_photo / 511.50) * 15.0)
    x[1] = math.floor((r_photo / 511.50) * 15.0)
    x[2] = int(avg_photo > last_photo)
    x[3] = math.floor(((temp - 20.0) / 10.0) * 15.0)
    x[4] = int(temp > last_temp)
    x[5] = math.floor(((uss - 2.0) / 100.0) * 5.0)
    x[6] = int(l_bump)
    x[7] = int(r_bump)

    # generalize
    for i in range(tile_num):
        rand = np.random.randint(-1, 2)
        gen_x_i[i, 0] = math.floor(bound_photo_data(l_photo + (photo_unit * rand)) / 511.50 * 15.0)
        rand = np.random.randint(-1, 2)
        gen_x_i[i, 1] = math.floor(bound_photo_data(r_photo + (photo_unit * rand)) / 511.50 * 15.0)
        rand = np.random.randint(-1, 2)
        gen_x_i[i, 2] = x[2]
        gen_x_i[i, 3] = math.floor(((bound_temp_data(temp + (temp_unit * rand)) - 20.0) / 10.0) * 15.0)
        gen_x_i[i, 4] = x[4]
        rand = np.random.randint(-1, 2)
        gen_x_i[i, 5] = math.floor(((bound_uss_data(uss + (uss_unit * rand)) - 2.0) / 100.0) * 5.0)
        gen_x_i[i, 6] = x[6]
        gen_x_i[i, 7] = x[7]

    # get q(s, a, w)
    for i in range(tile_num):
        q += w[i, gen_x_i[i, 0], gen_x_i[i, 1], gen_x_i[i, 2], gen_x_i[i, 3],
               gen_x_i[i, 4], gen_x_i[i, 5], gen_x_i[i, 6], gen_xp_i[i, 7], a]
        z_w[i, gen_x_i[i, 0], gen_x_i[i, 1], gen_x_i[i, 2], gen_x_i[i, 3],
            gen_x_i[i, 4], gen_x_i[i, 5], gen_x_i[i, 6], gen_xp_i[i, 7], a] += 1

    last_photo = avg_photo
    last_temp = temp

    # no movement and no vector updates during init.
    transmit_action = str(4) + '\n'
    serial.write(transmit_action.encode('utf-8'))
    serial.flush()
    serial.reset_input_buffer()

    while True:
        if serial.in_waiting > 5:

            data_list = byte_data_to_list()
            print(data_list)

            l_photo, r_photo, temp, uss = process_sensor_data(data_list)
            avg_photo = (l_photo + r_photo) / 2.0

            l_bump = int(data_list[6])
            r_bump = int(data_list[7])

            xp[0] = math.floor((l_photo / 511.50) * 15.0)
            xp[1] = math.floor((r_photo / 511.50) * 15.0)
            xp[2] = int(avg_photo > last_photo)
            xp[3] = math.floor(((temp - 20.0) / 10.0) * 15.0)
            xp[4] = int(temp > last_temp)
            xp[5] = math.floor(((uss - 2.0) / 100.0) * 5.0)
            xp[6] = int(l_bump)
            xp[7] = int(r_bump)
            print('xp: {}'.format(xp))

            # generalize
            for i in range(tile_num):
                rand = np.random.randint(-1, 2)
                gen_xp_i[i, 0] = math.floor(bound_photo_data(l_photo + (photo_unit * rand)) / 511.50 * 15.0)
                rand = np.random.randint(-1, 2)
                gen_xp_i[i, 1] = math.floor(bound_photo_data(r_photo + (photo_unit * rand)) / 511.50 * 15.0)
                rand = np.random.randint(-1, 2)
                gen_xp_i[i, 2] = xp[2]
                gen_xp_i[i, 3] = math.floor(((bound_temp_data(temp + (temp_unit * rand)) - 20.0) / 10.0) * 15.0)
                gen_xp_i[i, 4] = xp[4]
                rand = np.random.randint(-1, 2)
                gen_xp_i[i, 5] = math.floor(((bound_uss_data(uss + (uss_unit * rand)) - 2.0) / 100.0) * 5.0)
                gen_xp_i[i, 6] = xp[6]
                gen_xp_i[i, 7] = xp[7]

            # get reward based on xp
            r = get_reward(temp, l_bump, r_bump)
            avg_r += r

            # calculate policy given xp
            pi_theta = np.zeros(action_num)
            for i in range(action_num):
                for j in range(tile_num):
                    h_i = theta[j, gen_xp_i[j, 0], gen_xp_i[j, 1], gen_xp_i[j, 2], gen_xp_i[j, 3],
                                gen_xp_i[j, 4], gen_xp_i[j, 5], gen_xp_i[j, 6], gen_xp_i[j, 7], i]
                    pi_theta[i] = h_i
            norm = np.linalg.norm(pi_theta)
            pi_theta /= norm

            # get ap given policy
            ap = np.random.choice(np.arange(0, 4), p=pi_theta)
            print("action: ")
            print(ap)

            # send instruction to microcontroller
            transmit_action = str(ap) + '\n'
            serial.write(transmit_action.encode('utf-8'))
            serial.flush()
            serial.reset_input_buffer()

            # update eligibility trace of theta
            z_t = z_t * lambda_t
            for i in range(tile_num):
                z_t[i, gen_xp_i[i, 0], gen_xp_i[i, 1], gen_xp_i[i, 2], gen_xp_i[i, 3],
                    gen_xp_i[i, 4], gen_xp_i[i, 5], gen_xp_i[i, 6], gen_xp_i[i, 7], ap] += math.log(pi_theta[ap])

            # get q(s', a', w)
            qp = 0.0
            z_w = z_w * lambda_w
            for i in range(tile_num):
                qp += w[i, gen_xp_i[i, 0], gen_xp_i[i, 1], gen_xp_i[i, 2], gen_xp_i[i, 3],
                        gen_xp_i[i, 4], gen_xp_i[i, 5], gen_xp_i[i, 6], gen_xp_i[i, 7],  ap]
                z_w[i, gen_x_i[i, 0], gen_x_i[i, 1], gen_x_i[i, 2], gen_x_i[i, 3],
                    gen_x_i[i, 4], gen_x_i[i, 5], gen_x_i[i, 6], gen_xp_i[i, 7], ap] += 1

            # update TD error and average reward
            td_err = r - r_bar + qp - q
            r_bar += alpha_r * td_err

            avg_td_err += td_err
            avg_r += r_bar

            # update weight vector and theta vector
            w += alpha_w * td_err * z_w
            theta += alpha_t * td_err * z_t

            # write weight vector into memory
            if (t % 500 == 0):
                write_to_memory(0, w)
                write_to_memory(1, z_w)
                write_to_memory(2, z_t)
                write_to_memory(3, theta)
                print('t: {}'.format(t))
                print(
                    'memory stored ---------------------------------------------------------------------------------\n\n')

            if t % 10 == 0:
                write_to_record(4, (avg_r / 10.0))
                write_to_record(5, (avg_td_err / 10.0))
                r_bar, avg_td_err = 0.0, 0.0

            gen_x_i = gen_xp_i
            x = xp
            q = qp
            a = ap
            t += 1
