# API로 만드는 부분
from flask_restful import Resource, reqparse, fields, marshal_with
from com_cheese_api.cop.rvw.model.review_dao import ReviewDao
from com_cheese_api.cop.rvw.model.review_dto import ReviewDto, ReviewVo

review_fields = {
    'category': fields.Integer,
    'brand': fields.Integer,
    'cheese_name': fields.String,
    'review_title': fields.String,
    'review_detail': fields.String,
    'review_views': fields.Integer
}

class ReviewApi():
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        parser = self.parser
        parser.add_argument('category', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('brand', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('cheese_name', type=str, required=False, help='This field cannot be left blank')
        parser.add_argument('review_title', type=str, required=False, help='This field cannot be left blank')
        parser.add_argument('review_detail', type=str, required=False, help='This field cannot be left blank')
        parser.add_argument('review_views', type=str, required=False, help='This field cannot be left blank')
        args = parser.parse_args()
        article = ReviewDto(args['category'], args['brand'], args['cheese_name'],\
                            args['review_title'], args['review_detail'], args['review_views'])
        try:
            ReviewDao.save(review)
            return {'code' : 0, 'message' : 'SUCCESS'}, 200
        except:
            return {'message': 'An error occured inserting the review'}, 500

    @staticmethod
    def get(id):
        review = ReviewDao.find_by_id(id)
        if review:
            return review.json()
        return {'message': 'Review not found'}, 404

    @staticmethod
    def put(self, review, review_id):
        parser = self.parser
        parser.add_argument('review_no', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('category', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('brand', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('review_title', type=str, required=False, help='This field cannot be left blank')
        parser.add_argument('review_detail', type=str, required=False, help='This field cannot be left blank')
        args = parser.parse_args()
        review = ReviewVo()
        review.review_title = args['review_title']
        review.review_detail = args['review_detail']
        review.rev_id = args['rev_id']
        try:
            ReviewDao.update(review, review_id)
            return {'message': 'Review was Updated Successfully'}, 200
        except:
            return {'message': 'An Error Occured Updating the Review'}, 500


# 리뷰 리스트 
class Reviews(Resource):
    def get(self):
        return {'reivews': list(map(lambda review: review.json(), ReviewDao.find_all()))}
