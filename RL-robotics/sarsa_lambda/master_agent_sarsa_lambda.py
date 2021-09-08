import numpy as np
import serial
import random
import math

# servant arduino
serial = serial.Serial('/dev/ttyACM0', 115200, timeout=0.5)

# return true if increasing, false if decreasing or no change
def get_roc(_data, _record, _type):
    delta = 0.0
    for i in range(_record.size):
        delta += (_data - _record[i])
    delta /= float(_record.size)
    if _type == 0:  # photo data
        delta = (delta + 511.50) / 1023.0
    else:  # temp data
        delta = (delta + 10.0) / 20.0
    return delta > 0.5

def get_reward(_temp, _l_bump, _r_bump):
    r = 1.0 if (_temp >= 25.7 and _temp <= 26.2) else -1.0
    r = r - 5.0 if _l_bump else r
    r = r - 5.0 if _r_bump else r
    return r

def get_echo_index(_data):
    norm_data = (_data - 2.0) / 100.0
    return math.floor(((_data - 2.0) / 100.0) * 5.0)

def update_record(_record, _data):
    _record = np.roll(_record, -1)
    _record[-1:] = _data
    return _record

def byte_data_to_list():
    byte_data = serial.readline()
    data = byte_data[0:len(byte_data) - 2].decode("utf-8")
    data_list = np.array(data.split(), dtype=float)
    return data_list

def write_to_record(_mode, _avg_val):
    f_path = 'r.txt' if _mode else 'tde.txt'
    with open(f_path, "a") as record:
        record.write("\n{}".format(_avg_val))
    record.close()

def write_to_memory(_mode, _ndarray):
    f_path = 'w.txt' if _mode else 'z.txt'
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
    f_path = 'w.txt' if _mode else 'z.txt'
    with open(f_path, "r") as memory:
        data_list = np.array([])
        for i in range(4):
                data_row = np.array(memory.readline().split(), dtype=float)
                data_list = np.append(data_list, data_row)
        updated_ndarray = np.reshape(data_list, _ndarray.shape)
        return updated_ndarray

def get_bumper_state(_l, _r):
    if _l and _r:
        return 3
    elif _r:
        return 2
    elif _l:
        return 1
    return 0

def bound_temp_data(_data):
    _data = 29.9 if _data >= 29.9 else _data
    _data = 20.0 if _data < 20.0 else _data
    return _data

def bound_photo_data(_data):
    _data = 511.0 if _data >= 511.0 else _data
    _data = 0.0 if _data < 0.0 else _data
    return _data

def bound_uss_data(_data):
    _data = 101.5 if _data >= 101.5 else _data
    _data = 2.0 if _data < 2.0 else _data
    return _data

if __name__ == '__main__':

    eps = 0.01
    lambda_decay = 0.75
    alpha = 0.001
    gamma = 0.97
    feature_num = 7
    action_num = 5
    tile_num = 4

    # units per metric: width / tile_num
    photo_unit = 12.7875
    temp_unit = 0.5
    uss_unit = 5.0

    # w = [tilings][left photo][right photo][photo roc][temp][temp roc][sonic 90][bumper][action-num]
    #         4         10           10          2       10       2        5       4         5
    w = np.zeros((4, 15, 15, 2, 15, 2, 5, 4, 5))
    # or load from memory
    # w = read_from_memory(True, w)

    z = np.zeros((4, 15, 15, 2, 15, 2, 5, 4, 5))
    # or load from memory
    # z = read_from_memory(False, z)


    x = np.zeros(feature_num, dtype=int)
    xp = np.zeros(feature_num, dtype=int)
    gen_x_i = np.zeros((tile_num, feature_num), dtype=int)
    gen_xp_i = np.zeros((tile_num, feature_num), dtype=int)

    photo_record = np.zeros(3)
    temp_record = np.zeros(3)

    a, ap = 0, 0
    q, qp = 0.0, 0.0
    td_err = 0.0

    avg_r = 0.0
    avg_td_err = 0.0

    # initialize data records
    for i in range(3):

        while serial.in_waiting <= 5:
            continue

        data_list = byte_data_to_list()

        l_photo = data_list[0]
        r_photo = data_list[1]
        l_photo = bound_photo_data(l_photo)
        r_photo = bound_photo_data(r_photo)
        avg_photo = (l_photo + r_photo) / 2.0

        temp = data_list[2]
        temp = bound_temp_data(temp)

        l_bump = int(data_list[6])
        r_bump = int(data_list[7])

        if i == 2:

            watch30 = data_list[3]
            watch135 = data_list[4]
            watch90 = data_list[5]
            uss = bound_uss_data((watch90 * 0.5) + (watch30 * 0.25) + (watch135 * 0.25))

            x[0] = math.floor((l_photo / 511.50) * 15.0)
            x[1] = math.floor((r_photo / 511.50) * 15.0)
            x[2] = get_roc(avg_photo, photo_record, 0)
            x[3] = math.floor(((temp - 20.0) / 10.0) * 15.0)
            x[4] = get_roc(temp, temp_record, 1)
            x[5] = math.floor(((uss - 2.0) / 100.0) * 5.0)
            x[6] = get_bumper_state(l_bump, r_bump)

        update_record(photo_record, avg_photo)
        update_record(temp_record, temp)

        # no movement and no vector updates during init.
        transmit_action = str(6) + '\n'
        serial.write(transmit_action.encode('utf-8'))
        serial.flush()
        serial.reset_input_buffer()

    t = 0
    while True:

        if serial.in_waiting > 5:

            data_list = byte_data_to_list()
            print(data_list)

            l_photo = data_list[0]
            r_photo = data_list[1]
            l_photo = bound_photo_data(l_photo)
            r_photo = bound_photo_data(r_photo)
            avg_photo = (l_photo + r_photo) / 2.0

            temp = data_list[2]
            temp = bound_temp_data(temp)

            watch30 = data_list[3]
            watch135 = data_list[4]
            watch90 = data_list[5]
            uss = bound_uss_data((watch90 * 0.5) + (watch30 * 0.25) + (watch135 * 0.25))

            l_bump = int(data_list[6])
            r_bump = int(data_list[7])

            xp[0] = math.floor((l_photo / 511.50) * 15.0)
            xp[1] = math.floor((r_photo / 511.50) * 15.0)
            xp[2] = get_roc(avg_photo, photo_record, 0)
            xp[3] = math.floor(((temp - 20.0) / 10.0) * 15.0)
            xp[4] = get_roc(temp, temp_record, 1)
            xp[5] = get_echo_index(uss)
            xp[6] = get_bumper_state(l_bump, r_bump)
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
                rand = np.random.randint(-1, 2)
                gen_xp_i[i, 4] = xp[4]
                gen_xp_i[i, 5] = get_echo_index(bound_uss_data(uss + (uss_unit * rand)))
                gen_xp_i[i, 6] = xp[6]

            # collect reward
            td_err = get_reward(temp, l_bump, r_bump)
            avg_r += td_err

            for i in range(tile_num):
                td_err -= w[i, gen_x_i[i, 0], gen_x_i[i, 1], gen_x_i[i, 2],
                            gen_x_i[i, 3], gen_x_i[i, 4], gen_x_i[i, 5], gen_x_i[i, 6], a]
                z[i, gen_x_i[i, 0], gen_x_i[i, 1], gen_x_i[i, 2],
                  gen_x_i[i, 3], gen_x_i[i, 4], gen_x_i[i, 5], gen_x_i[i, 6], a] += 1

            # find action with the highest value for these features
            rand = random.random()
            if rand > eps:
                q_x = np.zeros(action_num)
                for i in range(tile_num):
                    for j in range(action_num):
                        q_x[j] += w[i, gen_xp_i[i, 0], gen_xp_i[i, 1], gen_xp_i[i, 2],
                                    gen_xp_i[i, 3], gen_xp_i[i, 4], gen_xp_i[i, 5], gen_xp_i[i, 6], j]
                qp = np.amax(q_x)
                ap_arr = np.where(q_x == qp)[0]
                ap = ap_arr[np.random.randint(ap_arr.size)]
            else:
                ap = np.random.randint(5)

            print("action: ")
            print(ap)

            transmit_action = str(ap) + '\n'
            serial.write(transmit_action.encode('utf-8'))
            serial.flush()
            serial.reset_input_buffer()

            for i in range(tile_num):
                td_err += gamma * w[i, gen_xp_i[i, 0], gen_xp_i[i, 1], gen_xp_i[i, 2],
                                    gen_xp_i[i, 3], gen_xp_i[i, 4], gen_xp_i[i, 5], gen_xp_i[i, 6], ap]

            # update weight vector for last time step
            w += alpha * td_err * z
            z = gamma * lambda_decay * z

            avg_td_err += td_err

            # write weight vector into memory
            if (t % 500 == 0):
                write_to_memory(True, w)
                write_to_memory(False, z)
                print('t: {}'.format(t))
                print(
                    'memory stored ---------------------------------------------------------------------------------\n\n')

            if t % 10 == 0:
                write_to_record(True, (avg_r / 10.0))
                write_to_record(False, (avg_td_err / 10.0))
                avg_r, avg_td_err = 0.0, 0.0

            photo_record = update_record(photo_record, avg_photo)
            temp_record = update_record(temp_record, temp)

            gen_x_i = gen_xp_i
            x = xp
            q = qp
            a = ap
            t += 1


