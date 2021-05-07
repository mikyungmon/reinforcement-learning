# Knowledge Distillation # 

  *Knowledge Distillation에 대해 알아보기 전에 "distillation(증류)"라는 단어에 대해 알아보면, distillation이란 액체 상태의 혼합물을 분리하는 방법을 말한다.*

  *쉽게 말해 a+b의 혼합물이 있으면 특정 기법을 사용하여 a 또는 b를 따로 추출해 내는 방법이다.*

  *-> 따라서 distillation이라는 용어를 인지할 때는 **'복잡하게 섞인 물체에서 필요로 하는 부분만 따로 추출하는 것'** 이라고 이해하면 좋다.*

- **Knowledge Distillation**은 NIPS 2014 에서 제프리 힌튼, 오리올 비니알스, 제프 딘 세 사람의 이름으로 제출된 "Distilling the Knowledge in a Neural Network" 라는 논문에서 제시된 개념이다.

- Knowledge Distillation 의 **목적**은 **"미리 잘 학습된 큰 네트워크(Teacher network) 의 지식을 실제로 사용하고자 하는 작은 네트워크(Student network) 에게 전달하는 것"** 이다.
  
  => 이 목적을 풀어서 설명하면 딥러닝 모델은 보편적으로 넓고 깊어서 파라미터 수가 많고 연산량이 많으면 feature extraction이 더 잘되고 그에 따라 classification이나 object detection같은 성능이 좋아진다. 
  
    하지만 딥러닝은 단순히 목적 성능이 좋은 모델이 좋은 모델이라고 하는 단계는 넘어섰으며 **computing resource(CPU & GPU)와 energy, memory 측면에서도 고려하여 성능을 평가해야한다.**
  
    예를 들어 핸드폰에서 딥러닝을 활용한 어플리케이션을 사용하고 싶은데, 몇 GB의 메모리를 필요로 하는 모델을 사용하려면 온라인 클라우드 서버 등에 접속해서 GPU 등의 자원을 사용해야 하지만, 이 모델을 충분히 작게 만들어서 핸드폰의 CPU 만으로도 계산이 가능하다면 여러가지 비용 측면에서 더 좋을 것이다.
 	
    **이렇게 Knowledge distillation 은 작은 네트워크도 큰 네트워크와 비슷한 성능을 낼 수 있도록, 학습과정에서 큰 네트워크의 지식을 작은 네트워크에게 전달하여 작은 네트워크의 성능을 높이겠다는 목적을 가지고 있다.**
     	
:bulb:   정리하면 1. 학습하고자 하는 Dataset D가 있을 때 Teacher Network T가 Dataset D를 먼저 학습한다. 

2) 그 후 Teacher Network보다 작은 규모의 Student Network S가 Teacher Network T를 활용하여 Dataset D를 학습한다. 

   **이 과정을 Distillation이라는 용어로 표현하며, 이는 Teacher Network T가 Student Network S에게 Dataset D에 관한 지식을 응축하여 전달하는 것이라고 할 수 있다.** 

   이렇게 학습된 Student Network S는 Teacher Network T 없이 Dataset D를 직접 학습한 S’보다 더 높은 성능을 보인다는 것이 여러 논문을 통해 확인되었다. 

   이것은 보다 큰 Neural Network인 Teacher Network T의 지식이 작은 Neural Network인 Student Network S로 Distillation 되었다고 하여 Knowledge Distillation이라고 하는 것이다.
   
## Architecture & Loss Function ##

![image](https://user-images.githubusercontent.com/66320010/117393863-fe515980-af2f-11eb-9ed7-aec205b49a3b.png)

- Loss function과 네트워크 구조를 같이 보면 이해하기 쉽다.

- Loss function의 **좌항**은 **Student network 의 분류 성능에 대한 loss**로, 그림 (가)에서 연두색 영역에 해당하며 Ground truth 와 Student 의 분류 결과와의 차이를 Cross entropy loss 로 계산한다. 

- Loss function의 **우항**은 **Teacher network 와 Student network 의 분류결과의 차이를 loss 에 포함**시키는 것으로, 그림 (가)에서 빨간색 영역에 해당하며 Teacher 와 Student 의 Output logit 을 Softmax로 변환한 값의 차이를 Cross entropy loss 로 계산한다. Teacher 와 Student 의 분류 결과가 같다면 작은 값을 취한다.

- 두 네트워크의 분류 결과를 비교하기 위해 hard label이 아닌 **soft label**을 사용하는데 다음 사진에서 왼쪽이 hard label 분류 결과이고 오른쪽이 soft label 분류 결과이다.

  ![image](https://user-images.githubusercontent.com/66320010/117395165-6a34c180-af32-11eb-91ac-0b7b38628ffe.png)
	
  ->  Soft label 을 곰곰이 생각해보면 입력 이미지에서 고양이와 개가 함께 가지고 있는 특징들이 어느정도 있었기 때문에 Dog class score 가 0.2 만큼 나왔다고 생각할 수 있다. 결과값을 Hard label 로 표현하면 이런 정보가 사라지게 된다.

- Hyperparameter 인 α, T 중에서 α 는 왼쪽항과 오른쪽항에 대한 가중치이다. α 가 크면 오른쪽항 Loss를 더 중요하게 보고 학습하겠다는 의미이며 T 는 Temperature 라고 부르는데 Softmax 함수가 입력값이 큰 것은 아주 크게, 작은 것은 아주 작게 만드는 성질을 완화해준다.

- 기존 Softmax 함수와 Temperature 를 사용한 Softmax 함수는 다음과 같이 표현된다.
  
  ![image](https://user-images.githubusercontent.com/66320010/117395678-5ccc0700-af33-11eb-8f84-52089630a0a1.png)
  
  예시를 들어 계산해보면 
  
  ![image](https://user-images.githubusercontent.com/66320010/117395720-73725e00-af33-11eb-931f-b3c3810e2de7.png)

  => 이와 같은 결과가 나오게 되는데 **Temperature 를 사용한 경우가 낮은 입력값의 출력을 더 크게 만들어주고 큰 입력값의 출력은 작게 만들어주는 것을 알 수 있다.** Temperature 를 사용하여, Soft label 을 사용하는 이점을 최대화해준다.

## 관련 논문 preview ##

**Diversity-driven knowledge distillation for financial trading using Deep Reinforcement Learning(2021)**

( Deep Reinforcement Learning을 사용하여 금융 거래를 위한 다양성 중심 지식 증류 )

논문 링크 : https://www.sciencedirect.com/science/article/pii/S0893608021000769
    
- **Abstract**

	심층 강화 학습 (RL)은 광범위한 작업을 위한 금융 거래 에이전트를 개발하는 데 점점 더 많이 사용되고 있다. 

	그러나 심층 에이전트를 최적화하기가 매우 어렵고 불안정하며, 특히 시끄러운 금융 환경이 거래 에이전트의 성능을 현저하게 방해한다.

	해당 논문에서는 **잘 알려진 신경망 증류(neural network distillation) 접근 방식을 기반으로 DRL 거래 에이전트의 훈련 신뢰성을 향상시키는 새로운 방법을 제시**한다. 

	제안된 접근 방식에서 교사 에이전트는 RL 환경의 여러 하위 집합에서 교육을 받았으므로 학습하는 정책을 다양화 한다.

	그런 다음 학생 에이전트는 훈련된 교사의 증류법을 사용하여 훈련 과정을 안내하여 솔루션 공간을 더 잘 탐색하는 동시에 교사 모델에서 제공하는 기존 정책 / 거래 전략을 '모방'한다. 

	제안된 방법의 효율성 향상은 다양한 통화에 대한 거래를 수행하도록 훈련된 교사의 다양한 앙상블을 사용하는데 있다. 

	이를 통해 가장 수익성이 높은 정책에 대한 공통된 견해를 학생에게 전달하여 시끄러운 금융 환경에서 교육 안정성을 더욱 향상시킬 수 있다. 

	수행된 실험에서 증류를 적용할 때 교사 모델을 다양화하도록 하면 최종 학생 에이전트의 성능을 크게 향상시킬 수 있다. 

	이 연구는 다양한 금융 거래 작업에 대한 광범위한 평가를 제공하여 이를 입증한다. 또한 제안된 방법의 일반성을 입증하기 위해 Procgen 환경을 사용하는 게임에서 별도의 제어 영역에서 추가 실험을 제공한다.
    
- **Proposed method**

	이 논문은 PPO (Proximal Policy Optimization) (Schulman et al., 2017)와 같은 Policy Gradient 기반 접근 방식을 사용하여 DRL 정책을 학습하는 데 중점을 둔다.

	제안된 방법은  Q-learning 기반 접근법과 같은 다른 방법에도 직접 적용될 수 있기 때문에 일반성을 잃지 않는다. 

	먼저 사용된 강화 학습 방법론이 간략하게 제시되고 그런 다음 제안된 증류 기반 방법을 제안된 다양성 기반 증류 방법과 함께 분석적으로 도출하고 논의한다.

	**1) Reinforcement learning methodolog( 강화 학습 방법론 )**
    	
	π (α | s)는 state s를 관찰하고 action α를 선택할 확률을 반환하는 DRL (Deep Reinforcement Learning) 모델의 출력을 나타낸다. 
	
	이 연구에서는 심층 RL 방법을 학습하기 위해 PPO (Proximal Policy Optimization) 방법을 사용한다.  
	
	처음에는 정책 매개 변수 단계에 대한 제약 조건을 적용한 trust 지역 정책 최적화(Schulman, Levine, Abbeel, Jordan, and Moritz, 2015)로 제안되었다. 
	
	![image](https://user-images.githubusercontent.com/66320010/117399665-0616fb00-af3c-11eb-9fde-009229f93615.png)
	
	사용된 제약 조건은 정책 업데이트 단계 사이의 KL (Kullback Leibler) 차이가 하이퍼 파라미터 ε 내에 남아 있음을 나타낸다. 이 식에서 πθ (·)는 매개 변수 θ인 정책이다.
	
	이전 단계 매개 변수는 θold로 표시됩니다. 이 방법은 인상적인 성능을 달성했지만 정책의 모든 업데이트 단계에 대한 제약 조건을 유지하는 것은 비현실적입니다. 유사한 결과를 제공하는 제안 된 휴리스틱 중 하나는 단계에 걸친 정책 매개 변수화 간의 행동 확률 비율을 활용하는 PPO (Proximal Policy Optimization)입니다. (2), 여기서 πθ (α | st)는 정책이 에이전트는 환경 statestat 시간 단계를 관찰합니다. ε 내에서 1 값 주변의 ratiort (θ)의 잘린 버전을 사용하여 정책 탐색을 매개 변수 pace에서 가까운 근처로 제한 할 수 있습니다. 잘린 정책 비율을 다음과 같이 정의합니다. (3), 여기서 ε은 정책 업데이트의 제약 범위. 최종 목적 함수는 다음과 같이 정의됩니다 : (4), 여기서 At는 이점이거나이 경우에는 궤적 내에서 astep의 일반화 된 이점 추정 (GAE) (Schulman, Moritz, Levine, Jordan, & Abbeel, 2015) (θ) 목표에 큰 영향을 미치기 위해 성과에 부정적인 영향을 주지만 목표에 대한 긍정적 인 이점의 효과를 제한하는 큰 부정적인 이점을 허용합니다.이 제약 조건은 과도하게 부정적인 보상을 피하는 위조적인 변경을 허용합니다. 그 결과 가장 명백한 보상 기울기에 대한 서두르는 정책 업데이트없이 긍정적으로 보상을받은 행동 경로를 보수적으로 탐색 할 수 있습니다. GAE를 계산하기 위해 각 시간 단계 t에 대한 시간차 (TD) 잔차를 다음과 같이 활용합니다. (5) 여기서 Rt는 에이전트가 시간 단계 t에서받는 보상이고 Vπ (st)는 다음과 같이 예측 된 값입니다. 현재 상태에 대한 정책 비판 st. 그런 다음 이점 Atis는 다음과 같이 정의됩니다. (6), 여기서 n은 롤아웃 내의 총 단계 수이고 t는 시간 단계입니다.
위에서 설명한 목표로 RL 에이전트를 교육하는 데 필요한 롤아웃을 축적하기 위해 경험 재생 메모리 (Mnih et al., 2015; Wawrzyński, 2009)가 사용됩니다. 경험 재생은 보상, 예측 된 상태 값 및 행동 확률과 같이 에이전트가 시뮬레이션 한 모든 궤적에 대한 관련 정보를 저장합니다. 경험 리플레이 메모리에 저장된이 정보는 에이전트의 행위자와 비평가 구성 요소를 구성하는 신경망을 훈련하기 위해 최적화 단계에서 샘플링됩니다.

위 내용 정리하고 3.2,3.3 추가예정
 
**Conclusions**

이 논문에서는 **교사 거래 에이전트에서 학생 에이전트로 지식을 이전하기위한 새로운 강화 학습 증류 체제가 제시**된다. 

**첫째**, 여러 훈련된 교사 에이전트의 지식을 학생 에이전트로 추출하면 간단한 시나리오에서도 학생 에이전트의 최종 거래 성능이 향상된다는 것이 실험적으로 입증되었다. 

교사 모델의 수를 늘리면 학생 모델의 성능이 향상되고 4 개의 교사 모델 이후의 수익이 감소한다. 

**둘째**, 증류 계수를 증가시키면 이 효과도 향상된다는 것을 보여준다. 

**마지막으로**, 우리는 사용 가능한 통화 쌍의 제약 하위 집합에서 교사를 훈련시켜 교사 간의 다양성을 강화하면 학생들의 성과가 크게 향상된다는 것을 실험적으로 검증했다. 

학생 증류가 제한 하위 집합만 활용 했더라도 개선 사항은 전체 데이터 세트에 걸쳐 있다. 

결과는 교사 에이전트의 이러한 다양화가 제약 데이터 세트 그룹 내에서 역테스트된 평균 PnL 측면에서뿐만 아니라 전체 데이터 세트의 평균 PnL 측면에서도 학생 성과를 향상시킬 수 있음을 보여준다.

미래에 탐구해야 할 흥미로운 영역은 여기에 제시된 통화 접근 방식과 함께 다양한 유형의 자산 (예 : 주식 및 채권)간에 다양한 에이전트의 상호 작용이다. 

이러한 자산 그룹은 서로 매우 다른 행동을 나타내므로 더 나은 증류를 목표로 하는 다각화 전략으로 유용할 수 있다.

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
