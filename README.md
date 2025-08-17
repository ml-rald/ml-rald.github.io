# ml-rald.github.io

ML랄드의 세번쨰 스터디를 위한 웹 공간입니다. ML랄드는 아카데미에서의 마지막 챌린지인 C6를 맞이하며, ML과 관련된 자유로운 아티클 쓰기 스터디를 진행합니다.

## 글쓰기 인스트럭션

1. 이슈를 올립니다.
2. 브랜치를 적당하게 만듭니다.  
    > ex> git switch -c posts/use-coreml-tools-to-convert-pytorch-model
3. content/posts 디렉토리에 적당한 이름으로 디렉토리를 하나 만들고 index.md를 작성합니다.  
    > ex> content/posts/use-coreml-tools-to-convert-pytorch-model/index.md

    ```md
    +++
    title = "CoreML Tools로 Pytorch 모델 변환해보기" # 제목 입력
    date = "2025-08-13T15:21:43+09:00" # 작성일시 입력. 연연연연-월월-일일T시시:분분:초초+09:00 형식
    #dateFormat = "2006-01-02"
    author = "Bob" # 작성자 입력
    authorTwitter = ""
    cover = ""
    tags = ["첫 포스트", "Core ML Tools"] # 태그 입력
    keywords = [""] # 키워드 입력. 없으면 쓰지 말기
    description = "" # 설명 입력
    showFullContent = false
    readingTime = false
    hideComments = false
    +++
    ```

    ## 제목은 이렇게 씁니다.

    ### 작은 제목은 이렇게 씁니다.

    본문은 이렇게 씁니다.

    이미지는 같은 디렉토리에 복사한 후 다음과 같이 씁니다.
    ![이미지에 대한 설명](./image.png)
3. 푸시하고 PR을 올립니다.
