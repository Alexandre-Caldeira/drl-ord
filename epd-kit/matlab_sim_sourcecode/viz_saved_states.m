
%%

for asnr = 15:-5:-5
% asnr = 5;
disp(asnr)
close all
clearvars -except asnr
% clc
% t2 = tic();
load(['snr_',num2str(asnr),'.mat'])
% toc(t2)
% 47.641746 seg 
% whos

% %%
% figure(1)
% subplot(131)
% stem(MSC(:,1))
% title('msc')
% subplot(132)
% stem(CSM(:,1))
% title('csm')
% subplot(133)
% stem(GFT(:,1))
% title('gft')
% 
% %%
% figure(2)
% stem(abs(SIGNALS(:,1)))
% 
% %%
% figure(3)
% ep = 1;
% estado = 2; %csm, gft, msc, t
% janela = 1;
% stem(all_freq_bins,states_vec(ep,:,estado,janela))
% 
% %%
% figure(4)
% subplot(121)
% histogram(reshape(msc_vec(:,noise_freq_bins(1),:),[],1))
% title('H0')
% subplot(122)
% histogram(reshape(msc_vec(:,signal_freq_bins(1),:),[],1))
% title('H1')
% 
% medida = 1; % csm, gft, msc, t
% 
% figure(5)
% subplot(121)
% histogram(10.^reshape(c_states_vec(:,find(all_freq_bins ==noise_freq_bins(1)),medida,:),[],1))
% title('H0')
% subplot(122)
% histogram(10.^reshape(c_states_vec(:,find(all_freq_bins==signal_freq_bins(1)),medida,:),[],1))
% title('H1')
% %%
% h0log = reshape(c_states_vec(:,find(all_freq_bins ==noise_freq_bins(1)),medida,:),[],1);
% h1log = reshape(c_states_vec(:,find(all_freq_bins==signal_freq_bins(1)),medida,:),[],1);
% 
% h0o = 10.^reshape(c_states_vec(:,find(all_freq_bins ==noise_freq_bins(1)),medida,:),[],1);
% h1o = 10.^reshape(c_states_vec(:,find(all_freq_bins==signal_freq_bins(1)),medida,:),[],1);
% 
% cv1 = quantile(h0log,1-0.05);
% cv2 = quantile(h0o,1-0.05);
% 
% td1 = round(100*sum(h1log<=cv2)/numel(h1o),4);
% td2 = round(100*sum(h1o>=cv2)/numel(h1o),4);
% 
% figure(6)
% histogram(h0log)
% hold on 
% histogram(h1log)
% xline(cv1)
% title(['log10, TPR = ',num2str(td1)])
% legend('h0','h1', ['cv1 = ',num2str(round(cv1,4))])
% 
% figure(7)
% histogram(h0o)
% hold on 
% histogram(h1o)
% xline(cv2)
% title(['original, TPR = ',num2str(td2)])
% legend('h0','h1', ['cv2 = ',num2str(round(cv2,4))])

%%

% save(['states_vec_snr_',num2str(snr_atual)],'states_vec')
% save(['msc_vec_snr_',num2str(snr_atual)],'msc_vec')
% save(['csm_vec_snr_',num2str(snr_atual)],'csm_vec')
% save(['gft_vec_snr_',num2str(snr_atual)],'gft_vec')
% save(['c_states_vec_snr_',num2str(snr_atual)],'c_states_vec')

end
% histogram(reshape(states_vec(:,18,1,:),[],1))
% histogram(reshape(states_vec(:,1,1,:),[],1))