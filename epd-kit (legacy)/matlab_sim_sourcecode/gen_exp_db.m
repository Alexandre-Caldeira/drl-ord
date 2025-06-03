%% SETUP
clearvars;
close all;
clc;


%% CONSTANTS

for cont_vol = 1:11
    for cont_int = 1:5
    t = tic();

    disp(['V,I = ',num2str(cont_vol),',',num2str(cont_int),' (',num2str(11),'/',num2str(5),')'])


    signal_freq_bins =  [82   90    84    86    88    90    92    94    96];
    noise_freq_bins = round(signal_freq_bins.*exp(1)/2)+5;
    all_freq_bins = [signal_freq_bins,noise_freq_bins];
    
    resolution = 10;
    max_length = 120;

    [SIGNALS,MSC,CSM, GFT,SNR_meas,MAG_freq, PHI_freq, c_states, states] = rlord_gen_log_states_exp(signal_freq_bins, ...
                                                noise_freq_bins, ...
                                                resolution,...
                                                cont_vol, ...
                                                cont_int);
    signals_vec = SIGNALS;
    snrmeas_vec = SNR_meas;
    magfreq_vec = MAG_freq;
    phifreq_vec = PHI_freq;
    msc_vec = MSC;
    csm_vec = CSM;
    gft_vec = GFT;
    c_states_vec = c_states;
    states_vec = states;

    toc(t)

    save(['signals_vec_vol_',num2str(cont_vol),'_int_',num2str(cont_int)],'signals_vec')
    save(['states_vec_vol_',num2str(cont_vol),'_int_',num2str(cont_int)],'states_vec')
    save(['snrmeas_vec_vol_',num2str(cont_vol),'_int_',num2str(cont_int)],'SNR_meas')
    save(['magfreq_vec_vol_',num2str(cont_vol),'_int_',num2str(cont_int)],'magfreq_vec')
    save(['phifreq_vec_vol_',num2str(cont_vol),'_int_',num2str(cont_int)],'phifreq_vec')
    save(['msc_vec_vol_',num2str(cont_vol),'_int_',num2str(cont_int)],'msc_vec')
    save(['csm_vec_vol_',num2str(cont_vol),'_int_',num2str(cont_int)],'csm_vec')
    save(['gft_vec_vol_',num2str(cont_vol),'_int_',num2str(cont_int)],'gft_vec')
    save(['c_states_vol_',num2str(cont_vol),'_int_',num2str(cont_int)],'c_states_vec')

    clearvars -except cont_vol cont_int
    end
end

