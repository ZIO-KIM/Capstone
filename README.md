# Capstone       

(2022-1) Capstone : 반려견 증상 기반 질환 진단 챗봇 <의사소통>       
- 2022학년도 상반기 세종대학교 Capstone Project의 일환으로 개발       
- 반려견의 종, 나이, 성별, 증상을 입력받아 예상 질병을 진단하는 AI Chatbot         
- 질병 진단은 ICD10 code를 기반으로 한 데이터를 활용 (이하 citation 참고) 
- SBert를 기반으로 한 유사도 검사 알고리즘을 활용, 증상을 한국어 자연어로 입력받고 단어형, 문장형 입력이 모두 가능하도록 설계           
- Python, Flask, JS를 활용한 예외처리 진행 및 웹 구현

- 호스팅 된 서비스는 아래의 주소를 통해 현재 접속 가능합니다. 
- http://aipetdoctor.pythonanywhere.com/

- 질병 진단을 위한 메인 데이터는 이하 논문을 참고하였습니다.  
Kim E, Choe C, Yoo JG, Oh S, Jung Y, Cho A, Kim S, Do YJ. 2018. Major medical causes by breed and life stage for dogs presented at veterinary clinics in the Republic of Korea: a survey of electronic medical records. PeerJ 6:e5161 https://doi.org/10.7717/peerj.5161
