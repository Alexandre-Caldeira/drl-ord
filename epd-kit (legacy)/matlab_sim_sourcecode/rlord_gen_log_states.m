function [SIGNALS,MSC,CSM, GFT,SNR_meas,MAG_freq, PHI_freq, c_states, states]= rlord_gen_log_states(signal_freq_bins, ...
                                                noise_freq_bins, ...
                                                snr, ...
                                                max_length,...
                                                resolution)
        
    res = checkForHarmonics(signal_freq_bins, noise_freq_bins);
    if res~= false
        error(['Signal and Noise frequencies contain harmonics in noise_freq=',...
            num2str(res(1))])
    end
    
    % fixed sampling rate...
    FS = 1000; 
    NFFT = FS; % rever isso aqui ! motivo/causas/impactos comp. 
    Njanelas = max_length;
    NpontosTotal = NFFT*Njanelas;    % Numero total de pontos de cada sinal
    tempo = (0:NpontosTotal-1)/FS;   % Vetor de tempo utilizado para gerar o sinal    
    
    % Gerando e observando sinais ruidosos no tempo e frequencia
    sigma_n = 2/NFFT;
    snr = 10^(snr()/10);
    
    % Constante a ser multiplicada ao ruido e ao sinal para configurar a relação sinal ruido desejada
    SNRs = sqrt(4*sigma_n*snr/NFFT);
    SNRn = sqrt(sigma_n);
    
    % cria matriz para receber a simulacao
    signals = nan(numel(signal_freq_bins), ...
                    NFFT*max_length);   
    
    % cria os sinais: (size = 8x1000xmax_length)
    for signal_idx = 1:numel(signal_freq_bins)
        SFREQ = signal_freq_bins(signal_idx)-1;
        signals(signal_idx ,:) = SNRs*sin(2*pi*SFREQ*tempo+rand()*2*pi);
    end
    
    ruido = randn(1,NpontosTotal);  % Gera um ruido gaussiano, teoricamente de variancia unitaria e media nula
    ruido = ruido-mean(ruido);      % Força a media nula
    ruido = ruido/std(ruido)*SNRn;  % Força a variancia desejada para o sinal
    signals = sum(signals,1)+ruido;
    signals = signals./std(signals);
    signals = reshape(signals,NFFT, Njanelas);
    
    % calcula transformada e estados contínuos:
    SIGNALS = fft(signals);
    SIGNALS = SIGNALS(1:floor(end/2)+1,:); % only half the FFT spectrum is valid
    f = FS/2*linspace(0,1,NFFT/2+1)'; % only half the FFT spectrum is valid
    
    MSC = nan(numel(f),max_length);
    CSM = nan(numel(f),max_length);
    GFT = nan(numel(f),max_length);
    SNR_meas = nan(numel(f),max_length);
    MAG_freq = nan(numel(f),max_length);
    PHI_freq = nan(numel(f),max_length);
                
    for idx_episodio = 1:max_length 
        
        % para cada frequencia em f
        for idx_f = 1:1:numel(f)
            
            
            M = idx_episodio;
            if M>40
                M=40;

                X_atual = SIGNALS(:,idx_episodio-M+1:idx_episodio);

            else

                X_atual = SIGNALS(:,1:idx_episodio);             
                
            end

            ymag = normalize(abs(X_atual));
            std_mov = std(ymag(idx_f,:),0,2);
            med_mov = mean(ymag(idx_f,:),2);
            SNR_meas(idx_f,idx_episodio) = real(-10*log10(med_mov./std_mov));

            
            GFT(idx_f,idx_episodio) = sum(abs(X_atual(idx_f,1:M,:)).^2)./...
                        (sum(abs(X_atual(idx_f,1:M,:)).^2)+sum(abs(X_atual(numel(f),1:M,:)).^2));
    
            c1_csm = (sum(cos(angle(X_atual(idx_f,:))))./M).^2;
            c2_csm = (sum(sin(angle(X_atual(idx_f,:))))./M).^2;
            CSM(idx_f,idx_episodio) = c1_csm+c2_csm;

            num_msc = abs(sum(X_atual(idx_f,:)))^2;
            den_msc= M*sum(abs(X_atual(idx_f,:)).^2);
            MSC(idx_f,idx_episodio) = num_msc/den_msc;
        end

        MAG_freq(:,idx_episodio) = abs(X_atual(:,end));
        PHI_freq(:,idx_episodio) = angle(X_atual(:,end));
        
    end
    
    % cria matriz dos estados continuos:
    c_states = nan(numel(signal_freq_bins)+numel(noise_freq_bins), ... % freqs
                    4,... % states
                    max_length); % windows

    all_freqs = [signal_freq_bins,noise_freq_bins];
    
    c_states(:,1,:) = log10(CSM(all_freqs,:));
    c_states(:,2,:) = log10(GFT(all_freqs,:)); 
    c_states(:,3,:) = log10(MSC(all_freqs,:));
    c_states(:,4,:) = repmat(100.*[1:max_length]./max_length,numel(all_freqs),1);
    c_states(:,5,:) = SNR_meas(all_freqs,:);
    c_states(:,6,:) = MAG_freq(all_freqs,:);
    c_states(:,7,:) = PHI_freq(all_freqs,:);
    
    % discretiza a matriz de estados
    resolucao = resolution; % 10 niveis discretos para cada estado
    states =  nan(size(c_states));

    for idx_freq = 1:numel(all_freqs)
        for idx_state = 1:size(c_states,2)
            values = c_states(idx_freq,idx_state,:);
            min_val = min(c_states(:,idx_state,:),[],'all');
            max_val = max(c_states(:,idx_state,:),[],'all');
    
            states(idx_freq,idx_state,:) = discretize_val( ...
                                            values, min_val, max_val, resolucao);
        end
    end

end