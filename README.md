# Spaceship-Titanic
## Competition Kaggle
# Описание набора данных
##### ссылка на streamlit: https://spaceship-titanic-ylly6zajyosp7wxuaivomq.streamlit.app/
##### данные взяты с https://www.kaggle.com/competitions/spaceship-titanic
В этом соревновании задача - предсказать, был ли пассажир перенесен в альтернативное измерение во время столкновения космического корабля "Титаник" с пространственно-временной аномалией. Чтобы помочь вам сделать эти прогнозы, вам предоставляется набор личных записей, извлеченных из поврежденной компьютерной системы корабля.

- Описания файлов и полей данных
    - 1.train.csv - Личные записи примерно двух третей (~ 8700) пассажиров, которые будут использоваться в качестве обучающих данных.
    - 2.PassengerId - Уникальный идентификатор для каждого пассажира. Каждый идентификатор принимает форму, gggg_pp где gggg указывается группа, с которой путешествует пассажир, и pp его номер в группе. Люди в группе часто являются членами семьи, но не всегда.
    - 3. HomePlanet - Планета, с которой вылетел пассажир, обычно планета его постоянного проживания.
    - 4. CryoSleep - Указывает, решил ли пассажир погрузиться в анабиоз на время полета. Пассажиры в криосне прикованы к своим каютам.
    - 5. Cabin - Номер каюты, в которой находится пассажир. Принимает форму, deck/num/sideгде side может быть либо P для левого, либо S для правого борта.
    - 6. Destination - Планета, на которой высадится пассажир.
    - 7. Age - Возраст пассажира.
    - 8. VIP - Оплатил ли пассажир специальное VIP-обслуживание во время путешествия.
    - 9. RoomServiceFoodCourt, ShoppingMallSpa, VRDeck - Сумма, которую пассажир выставил за пользование всеми многочисленными удобствами класса люкс на "Космическом корабле "Титаник"".
    - 10. Name - Имя и фамилия пассажира.
    - 11. Transported - Был ли пассажир перенесен в другое измерение. Это цель, столбец, который вы пытаетесь предсказать.
    
- test.csv - Личные записи для оставшейся трети (~ 4300) пассажиров, которые будут использоваться в качестве тестовых данных. Ваша задача - предсказать стоимость Transported для пассажиров в этом наборе.
- <strong>sample_submission.csv</strong> - файл отправки в правильном формате.
    - PassengerId - Идентификатор для каждого пассажира в тестовом наборе.
    - Transported - Цель. Для каждого пассажира предскажите либо True, либо False.
