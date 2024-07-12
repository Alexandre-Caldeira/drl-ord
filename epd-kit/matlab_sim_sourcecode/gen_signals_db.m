%% SETUP
clearvars;
close all;
clc;

% addpath('C:\PPGEE\EEE935 Aprendizado por Reforço\quenaz')

%% CONSTANTS

% NsinaisTotal = 2500; %Número de sinais totais que serão gerados. Obrigatóriamente múltiplo de 100
% Njanelas = 120;%[30 40 60];% 90 120];
% NFFT = 1000;
% SFREQ = 81;
% FS = 1000;
% %% SCRIPT
% % Elapsed time is 238.565675 seconds. (PER SNR)
% % t = tic();
% % toc(t)
% M = Njanelas;
% % snr_atual = 10;
% % snrs = 15:-0.5:-30;
% snrs = 15:-5:-30;
% for isnr = 1:numel(snrs)
%     disp(isnr)
%     snr_atual = snrs(isnr);
%     SNRfun = @() snr_atual+0.5*randn;
%     sinal = gen_signals(SNRfun, FS, SFREQ, NFFT, NsinaisTotal, M);
%     var = 'sinal';
%     caminho_nome_arquivo = [var,'_snr_',num2str(snr_atual)];
%     save(caminho_nome_arquivo, var, '-v7.3')
% end
% % var = 'csm';
% % caminho_nome_arquivo = [var,'_snr_pos_',num2str(snr_atual)];
% % save(caminho_nome_arquivo, var, '-v7.3')
% % var = 'msc';
% % caminho_nome_arquivo = [var,'_snr_pos_',num2str(snr_atual)];
% % save(caminho_nome_arquivo, var, '-v7.3')

%% SCRIPT 2

% 10000/3600 = 2.7778 hrs por SNR
% snrs = 15:-0.5:-30;
% snrs = 15:-5:-30;
snrs = 5:-2.5:-5;

for isnr = 1:numel(snrs)
    t = tic();
    snr_atual = snrs(isnr);
    disp(['SRN = ',num2str(snr_atual),' (',num2str(isnr),'/',num2str(numel(snrs)),')'])
    SNRfun = @() snr_atual+2.5*randn;

    NsinaisTotal = 2500;
    
    signal_freq_bins =  [82   90    84    86    88    90    92    94    96];
    noise_freq_bins = round(signal_freq_bins.*exp(1)/2)+5;
    all_freq_bins = [signal_freq_bins,noise_freq_bins];
    
    resolution = 10;
    max_length = 120;
    
    signals_vec = zeros(NsinaisTotal,501, max_length);
    msc_vec = zeros(NsinaisTotal,501, max_length);
    snrmeas_vec = zeros(NsinaisTotal,501, max_length);
    magfreq_vec = zeros(NsinaisTotal,501, max_length);
    phifreq_vec = zeros(NsinaisTotal,501, max_length);
    csm_vec = zeros(NsinaisTotal,501, max_length);
    gft_vec = zeros(NsinaisTotal,501, max_length);
    c_states_vec = zeros(NsinaisTotal,length(all_freq_bins),7, max_length);
    states_vec = zeros(NsinaisTotal,length(all_freq_bins),7, max_length);

    for i =1:NsinaisTotal
        if rem(i,500)==0
            disp(['Sim atual = ',num2str(i)])
        end
        [SIGNALS,MSC,CSM, GFT,SNR_meas,MAG_freq, PHI_freq, c_states, states] = rlord_gen_log_states(signal_freq_bins, ...
                                                    noise_freq_bins, ...
                                                    SNRfun, ...
                                                    max_length,...
                                                    resolution);
        signals_vec(i,:,:) = SIGNALS;
        snrmeas_vec(i,:,:) = SNR_meas;
        magfreq_vec(i,:,:) = MAG_freq;
        phifreq_vec(i,:,:) = PHI_freq;
        msc_vec(i,:,:) = MSC;
        csm_vec(i,:,:) = CSM;
        gft_vec(i,:,:) = GFT;
        c_states_vec(i,:,:,:) = c_states;
        states_vec(i,:,:,:) = states;
    end
    toc(t)

    save(['signals_vec_snr_',num2str(snr_atual)],'signals_vec')
    save(['states_vec_snr_',num2str(snr_atual)],'states_vec')
    save(['snrmeas_vec_snr_',num2str(snr_atual)],'SNR_meas')
    save(['magfreq_vec_vec_snr_',num2str(snr_atual)],'magfreq_vec')
    save(['phifreq_vec_snr_',num2str(snr_atual)],'phifreq_vec')
    save(['msc_vec_snr_',num2str(snr_atual)],'msc_vec')
    save(['csm_vec_snr_',num2str(snr_atual)],'csm_vec')
    save(['gft_vec_snr_',num2str(snr_atual)],'gft_vec')
    save(['c_states_vec_snr_',num2str(snr_atual)],'c_states_vec')

    clearvars -except isnr snrs
end

