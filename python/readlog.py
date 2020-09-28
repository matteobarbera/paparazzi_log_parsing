import pdb
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np


class rotorcraft_fp_class:
    """Rotorcraft Class"""
    t = 0

    def f(self):
        return 'hello world'


def time_align_data(time1, v1, time2, v2):
    """
    Takes two vectors with measurements of different length (v1, v2) - recorded at different time steps (time1, time2)
    Returns the vectors and times with length equal to the maximum length, linearly interpolating values in v
    """

    # 1) join the two time lists into one series
    # 2) per time step see if it exists in vector v1 / v2
    # 3) if it does not exist, insert a linearly interpolated element

    # 1)
    # first take out double times:
    time1, inds1 = np.unique(time1, return_index=True);
    v1 = v1[inds1];
    time2, inds2 = np.unique(time2, return_index=True);
    v2 = v2[inds2];
    # then concatenate, get all unique times, and sort:
    times = np.concatenate([time1, time2]);
    times = np.unique(times);
    sorted_times = np.sort(times, axis=None);
    # go over all elements:
    for t in sorted_times:
        # 2) / 3)
        [v1, time1] = check_for_t(v1, time1, t);
        [v2, time2] = check_for_t(v2, time2, t);

    return [time1, v1, time2, v2];

def check_for_t(v, time, t):
    """ 
    If timestep t does not exist in the vector "time", add it,
    and enter an interpolated value from vector v.
    """
    pad_value = 0;
    # 2)
    inds = np.where(time == t);
    if inds[0].size == 0:
        # 3)
        # time step does not exist:
        inds_l = np.where(time > t);
        inds_s = np.where(time < t);
        if inds_l[0].size == 0:
            # larger than the other times, append at the back:
            v = np.append(v, pad_value);
            time = np.append(time, t);      
        elif inds_s[0].size == 0:
            # smaller than the other times, insert at the front:
            v = np.insert(v, 0, pad_value);
            time = np.insert(time, 0, t);
        else:
            # value of the largest smaller time step
            v_ts = v[inds_s[0][-1]];
            # value of the smallest larger time step
            v_tl = v[inds_l[0][0]];
            # t smaller / larger
            t_s = time[inds_s[0][-1]];
            t_l = time[inds_l[0][0]];
            # linear interpolation:
            alpha = (t - t_s) / (t_l - t_s);
            new_val = alpha * v_tl + (1-alpha) * v_ts;
            # insert the values at the right place:
            v = np.insert(v, inds_l[0][0], new_val);
            time = np.insert(time, inds_l[0][0], t);

    # return the possibly modified vector:
    return [v, time];
                

def linear_fit(A, B, BIAS = False, PRIOR=True, alpha_prior=1.0):
    """
    Perform a linear fit, A x = B, returning x
    """

    # Add a bias or not:
    if(BIAS):
      A = np.concatenate((A, np.ones((A.shape[0], 1))), axis=1);

    if(PRIOR):
      # Prior:
      # weights = inv(F_training' * F_training + alpha * eye(size(F_training, 2))) * F_training' * H_training;
      x = np.dot(np.linalg.inv(np.dot(np.transpose(A), A) + alpha_prior * np.diag(np.ones(A.shape[1]))), np.dot(np.transpose(A), B));

    else:
      # No prior:
      x = np.linalg.lstsq(A, B)[0];

    return x;

def get_windowed_cov_div(S, window_size, delta_sample):
    
    N = S.shape[0] - window_size - delta_sample;
    cov_div = np.zeros([S.shape[0],1]);
    for i in range(N):
        
        A = S[i+delta_sample:i+window_size+delta_sample];
        B = S[i:i+window_size];
        cov_div[i+window_size+delta_sample] = get_cov_div(A,B);
    
    return cov_div;

def get_cov_div(A,B):
    if(A.shape != B.shape):
        pdb.set_trace();
    C = np.cov(A,B);
    return C[0,1];

def plot_log_file(filename, nr):

    gps_int = []
    rotorcraft_fp = []
    rotorcraft_fp_min = []
    rotorcraft_status_txt = []
    motor = []
    imu_mag_raw_txt = []
    imu_gyro_scaled_txt = []
    divergence_landing_txt = []
    optic_flow_txt = []
    textons_txt = [];


    tags = {}
    ids = {}
    autopilot_version = {}

    #pdb.set_trace();

    with open(filename) as f:
        for line in f:
            
            d = line.strip().split(" ")
            if len(d) > 2:
                tags[d[2]] = 1;
                ids[d[1]] = 1;
            line = line.replace("replay", "1000")
            #if "replay" in line:
            #    return "REPLAY";
            #elif "NPS_" in line:
            #    return "SIM";
            #el
            if "GPS_INT" in line:
                gps_int.append(line.replace("GPS_INT ", ""))
            elif "MOTOR" in line:
                motor.append(line.replace("MOTOR ", ""))
            elif "IMU_MAG_RAW" in line:
                imu_mag_raw_txt.append(line.replace("IMU_MAG_RAW ", ""))
            elif "IMU_GYRO_SCALED" in line:
                imu_gyro_scaled_txt.append(line.replace("IMU_GYRO_SCALED ", ""))
            elif "ROTORCRAFT_STATUS" in line:
                rotorcraft_status_txt.append("".join(line.split("ROTORCRAFT_STATUS ")))
            elif "ROTORCRAFT_FP_MIN" in line:
                rotorcraft_fp_min.append("".join(line.split("ROTORCRAFT_FP_MIN ")))
            elif "ROTORCRAFT_FP" in line:
                rotorcraft_fp.append("".join(line.split("ROTORCRAFT_FP ")))
            elif "AUTOPILOT_VERSION" in line:
                autopilot_version[line.strip().split("AUTOPILOT_VERSION")[1]] = 1
            elif "OPTIC_FLOW_EST" in line:
                optic_flow_txt.append(line.replace("OPTIC_FLOW_EST", ""));
            elif "DIVERGENCE" in line:
                divergence_landing_txt.append(line.replace("DIVERGENCE", ""));
            elif "TEXTONS" in line:
                textons_txt.append(line.replace("TEXTONS", ""));
        
#         <message name="DIVERGENCE" id="224">
#         <field name="divergence" type="float"> vertical velocity / height from optitrack (unit = 1/sec)</field>
#         <field name="divergence_vision" type="float"> vertical velocity / height from vision (unit = 1/sec)</field>
#         <field name="normalized_thrust" type="float"> thrust / max thrust paparazzi (-)</field>
#         <field name="cov_div" type="float"> covariance of divergence and thrust, or past divergence depending on the mode (-)</field>
#         <field name="pstate" type="float"> gain state in adaptive gain control: indicative of height (-) </field>
#         <field name="pused" type="float"> gain used for control, includes the effect of the p-gain of adaptive control (-) </field>
#         <field name="sonar" type="float"> measurement from the sonar (mm)</field>
#         </message>
            

    #print(" - ", ", ".join(sorted(ids.keys())))
    #print(" - ", ", ".join(sorted(tags.keys())))
    #print(" - ", ", ".join(autopilot_version.keys()))

    #g = np.fromstring(s, dtype=None, sep=' ')
    #g = np.genfromtxt(StringIO(s), delimiter=' ', dtype=None)

    g = np.empty([0,0]);
    if len(gps_int) > 0:
        g = np.loadtxt(StringIO(u"".join(gps_int)), delimiter=' ', dtype=None)
        
    r = np.empty([0,0])
    if len(rotorcraft_fp) > 0:
        r = np.loadtxt(StringIO(u"".join(rotorcraft_fp)))
             
    rotorcraft_status = np.empty([0,0])
    if len(rotorcraft_status_txt) > 0:
        rotorcraft_status = np.loadtxt(StringIO(u"".join(rotorcraft_status_txt)))
             
    m = np.empty([0,0])
    if len(motor) > 0:
        m = np.loadtxt(StringIO(u"".join(motor)))
             
    imu_mag_raw = np.empty([0,0])
    if len(imu_mag_raw_txt) > 0:
        imu_mag_raw = np.loadtxt(StringIO(u"".join(imu_mag_raw_txt)))
             
    imu_gyro_scaled = np.empty([0,0])
    if len(imu_gyro_scaled_txt) > 0:
        imu_gyro_scaled = np.loadtxt(StringIO(u"".join(imu_gyro_scaled_txt)))

    divergence_landing = np.empty([0,0])
    if len(divergence_landing_txt) > 0:
        divergence_landing = np.loadtxt(StringIO(u"".join(divergence_landing_txt)))

    optic_flow = np.empty([0,0])
    if len(optic_flow_txt) > 0:
        optic_flow = np.loadtxt(StringIO(u"".join(optic_flow_txt)))

    textons = np.empty([0,0])
    if len(textons_txt) > 0:
        textons = np.loadtxt(StringIO(u"".join(textons_txt)))


    #print(g)
    #print(r)

    # whether to do interactive plotting (e.g., when using pdb.set_trace();)
    #plt.ion();

    if textons.size > 0:
        # learn weights from the textons:
        # structure texton message: [textons gain cov_div]
        n_textons = textons.shape[1] - 2;         
        n_samples = textons.shape[0];
        texton_features = textons[:, 0:-2];
        gains = textons[:, -1];
        cov_divs = textons[:, -2];
        n_training = int(0.8 * n_samples);
        features_training = features[0:n_training, :];
        gains_training = gains[0:n_training];
        features_test = features[n_training+1:];
        gains_test = gains[n_training+1:];

        # different learning methods:
        weights_ML_no_bias = linear_fit(features_training, gains_training, False, False);
        weights_ML_bias = linear_fit(features_training, gains_training, True, False);
        weights_MAP_prior1 = linear_fit(features_training, gains_training, True, True, 1.0);
        weights_MAP_prior10 = linear_fit(features_training, gains_training, True, True, 10.0);

        # test them on the test set:
        test_error = np.zeros(4);
        gain_estimates = np.dot(features_test, weights_ML_no_bias);        
        test_error[0] = np.mean(np.abs(gains_test - gain_estimates));
        gain_estimates = np.dot(features_test, weights_ML_bias);        
        test_error[1] = np.mean(np.abs(gains_test - gain_estimates));
        gain_estimates = np.dot(features_test, weights_MAP_prior1);        
        test_error[2] = np.mean(np.abs(gains_test - gain_estimates));
        gain_estimates = np.dot(features_test, weights_MAP_prior10);        
        test_error[3] = np.mean(np.abs(gains_test - gain_estimates));

        # print test set errors:
        for i in range(4):
          print('Error %d = %f' % (i, test_error[i]));

        # save the best weights:

    if divergence_landing.size > 0:

        N = 30; # smoothing window size:

        # get and preprocess the measurements:
        time_steps = divergence_landing[:,0];

        # the factor multiplied with divergence_vision_dt should be put in the optic_flow_landing.c code:
        div_vision_dt = divergence_landing[:,3]; # div_vision_dt # 0.00008*
        dt_time_steps = time_steps;

        div_vision = divergence_landing[:,2]; # filtered and scaled

        # if filtering in Python, overwrite div_vision:
        filter_in_python = False;
        if(filter_in_python):
            max_change = 0.30;
            lp_factor = 0.75;
            div_lp = 0.0;
            for i in range(div_vision_dt.size):
                new_div = div_vision_dt[i];
                if(abs(new_div-div_lp) > max_change):
                    if(new_div > div_lp):
                        new_div = div_lp + max_change;
                    else:
                        new_div = div_lp - max_change;                        

                div_lp = lp_factor * div_lp + (1-lp_factor) * new_div;
                div_vision[i] = div_lp;


        div_raw = optic_flow[:, -3]; # 0.025 * 
        time_div_raw = optic_flow[:, 0];
        div_vision = np.convolve(div_vision, np.ones((N,))/N, mode='same');
        height = divergence_landing[:,8];

        # smooth the height:
        height = np.convolve(height, np.ones((N,))/N, mode='same');
        n_steps = 10; # to get a less noisy estimate we don't take two subsequent samples, but take a value further back.
        dt = time_steps[n_steps:-1] - time_steps[0:-n_steps-1];
        velocity = np.divide(height[n_steps:-1] - height[0:-n_steps-1], dt);
        div_truth = np.divide(velocity, height[n_steps:-1]);

        # rotorcraft status:
        time_rcs = rotorcraft_status[:,0];
        ap_mode = rotorcraft_status[:,8];

        # get the GPS height:
        time_gps = g[:,0];
        height_gps = g[:,8] / 1000.0;
        [new_time_steps, height, time_gps, height_gps] = time_align_data(time_steps, height, time_gps, height_gps);
        # also align the vision estimates:
        [new_time_steps, height, time_steps, div_vision] = time_align_data(new_time_steps, height, time_steps, div_vision);

        # get the velocity from GPS:
        dt_gps = time_gps[n_steps:-1] - time_gps[0:-n_steps-1];
        # smooth the height:
        height_gps = np.convolve(height_gps, np.ones((N,))/N, mode='same');
        velocity_gps = np.divide(height_gps[n_steps:-1] - height_gps[0:-n_steps-1], dt_gps);
        div_truth_gps = np.divide(velocity_gps, height_gps[n_steps:-1]);

        # whether to fit the measured vision divergence to the ground truth divergence (a form of calibration):
        make_fit = False;

        if make_fit:
            # make a fit from 301 to 317 seconds:
            inds_s = np.where(time_steps <= 110);
            inds_l = np.where(time_steps >= 150);
            d_size = inds_l[0][0] - inds_s[0][-1];
            output_div = np.zeros([d_size, 1]);
            output_div[:,0] = div_truth_gps[inds_s[0][-1]:inds_l[0][0]];
            input_div = np.zeros([d_size, 1]);
            input_div[:,0] = div_vision[inds_s[0][-1]:inds_l[0][0]];

            # just scale:
            # do linear least squares to map the vision divergence to the ground truth values:
            f_resp = np.linalg.lstsq(input_div, output_div);
            scale = f_resp[0][0];
            res1 = f_resp[1][0];

            # linear_fit(A, b, BIAS = False, PRIOR=True, alpha_prior=1.0);
            # linear_fit(input_div, 2.0*input_div+1.0, True);

            # fit scale and bias:
            bias = np.ones([input_div.size, 1]);
            inputs = np.zeros([d_size, 2]);
            inputs[:,0] = input_div[:,0];
            inputs[:,1] = bias[:,0];
            f_resp = np.linalg.lstsq(inputs, output_div);
            pars = f_resp[0];
            res2 = f_resp[1][0];

            # print the results:
            print("Just a scale: %f, error = %f\n" % (scale, res1));
            print("Scale and bias: %f, %f, error = %f\n" % (pars[0], pars[1], res2));

        # plot vision divergence and ground truth divergence in the same plot:
        f = plt.figure();
        # plt.plot(divergence_landing[:,0], divergence_landing[:,2]);
        # plt.plot(time_gps[n_steps:-1], div_truth_gps);
        
        if make_fit:
            plt.plot(time_gps[n_steps:-1], div_truth_gps, time_steps[n_steps:-1], scale * div_vision[n_steps:-1], time_rcs, ap_mode); 
        else:
            plt.plot(time_gps[n_steps:-1], div_truth_gps, time_steps[n_steps:-1], div_vision[n_steps:-1], dt_time_steps, div_vision_dt, time_div_raw, div_raw); # time_rcs, ap_mode,
        plt.legend(['div truth', 'div vision low-pass', 'div vision dt', 'raw measurements'])
        plt.show();
        
        # determining the window size and delta time for the covariance divergence oscillation detection with shifted time window:
        mean_time_step = np.mean(dt/n_steps);
        window_size = 100;
        delta_sample = 25;
        print('Covdiv 1: Window size in seconds: %f, delta t in seconds: %f\n' % (window_size*mean_time_step, delta_sample*mean_time_step))
        cov_div_1 = get_windowed_cov_div(divergence_landing[:,2], window_size, delta_sample);
        delta_sample = 20;
        print('Covdiv 2: Window size in seconds: %f, delta t in seconds: %f\n' % (window_size*mean_time_step, delta_sample*mean_time_step))
        cov_div_2 = get_windowed_cov_div(divergence_landing[:,2], window_size, delta_sample);
        plt.plot(divergence_landing[:,0], divergence_landing[:,2], divergence_landing[:,0], 10*cov_div_1, divergence_landing[:,0], 10*cov_div_2);
        plt.show();
        
        
    ###########################
    # Guess File Type:

    fig = plt.figure(nr)

    #####
    # RAW
    if imu_mag_raw.size > 0:
        ax = fig.gca( projection='3d')
        ax.plot(imu_mag_raw[:,2], imu_mag_raw[:,3], imu_mag_raw[:,4],label='imu_mag_raw')
        ax.legend()
        #ax.grid()
        
    #############
    # ONBOARD LOG
    if imu_gyro_scaled.size > 0:
        TURN_RATE_FRAC = 2^12
        plt.plot(imu_gyro_scaled[:,0], imu_gyro_scaled[:,2]/TURN_RATE_FRAC)
        plt.plot(imu_gyro_scaled[:,0], imu_gyro_scaled[:,2]/TURN_RATE_FRAC)
        plt.plot(imu_gyro_scaled[:,0], imu_gyro_scaled[:,3]/TURN_RATE_FRAC)
        plt.grid()
        plt.title(filename)
        return "RAW, " + ", ".join(sorted(ids.keys()))


    ############
    # ROTORCRAFT
    elif g.size > 8:
        inflight = 0;
        if rotorcraft_status.size > 28:
            np.sum(rotorcraft_status[:,8])
        title = "ROTOR, " + ", ".join(sorted(ids.keys())) + ", " + str(inflight)
        if "NPS_SENSORS_SCALED" in tags:
            title = "SIMULATION, " + ", ".join(sorted(ids.keys()))  + ", " + str(inflight)

        plt.subplot(2, 2, 1)
        plt.plot(g[:,0], g[:,8]/1000.0)
        plt.title(filename)
        plt.ylabel('alt [m]')
        plt.grid()
        plt.subplot(2, 2, 2)
        if m.size > 28:
            plt.plot(m[:,0], m[:,2])
        plt.ylabel('rpm [1/m]')
        plt.xlabel('time [s]')
        plt.grid()
        plt.subplot(2, 2, 3)
        plt.plot(r[:,0], r[:,16])
        plt.grid()
        plt.title(title)
        plt.subplot(2, 2, 4)
#        print(rotorcraft_status.size > )
        if rotorcraft_status.size > 28:
           plt.plot(rotorcraft_status[:,0], rotorcraft_status[:,12]/10.0)
        plt.grid()
        plt.ylabel('vbat [V]')

        return title
 

    return "NODATA"

if __name__ == "__main__":
    print('Start')
    #filename = '16_08_02_Heemskerk/16_08_02__15_13_52.data'
    #filename = '16_04_11_monday/16_04_11__18_24_17.data'
    #filename = '16_08_01 Kalmthout auto2\sd_kaart/16_08_01__22_33_34_SD_no_GPS.data'
    filename = '20_03_04__15_53_09_SD.data'

    p = plot_log_file(filename,1)

    print('Ready')

    p.show()

    #pp = PdfPages('test.pdf')
    #pp.savefig(1)
    #pp.close()
    

    

