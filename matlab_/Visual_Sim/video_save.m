v = VideoWriter('sim_output.mp4', 'MPEG-4'); % MP4 형식으로 저장
v.FrameRate = 30; % 초당 프레임 수 설정
open(v);

for i = 1:200  % 200프레임 녹화 (원하는 만큼 증가 가능)
    frame = getframe(gcf);  % 현재 Mechanics Explorer 창 캡처
    writeVideo(v, frame);   % 비디오에 프레임 추가
    pause(0.05);            % 시뮬레이션과 동기화
end

close(v);
disp('비디오 저장 완료! sim_output.mp4');
