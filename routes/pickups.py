from flask import Blueprint, request, redirect, url_for, session
from models import db, Pickup

pickups_bp = Blueprint('pickups', __name__)

@pickups_bp.route('/update_pickup/<int:pickup_id>', methods=['POST'])
def update_pickup(pickup_id):
    action = request.form.get('action')
    pickup = Pickup.query.get(pickup_id)

    if not pickup:
        return "Pickup not found", 404

    if action == 'accept':
        pickup.status = 'Accepted'
        pickup.collector_id = session.get('id')
    elif action == 'decline':
        pickup.status = 'Declined'
    elif action == 'complete':
        pickup.status = 'Completed'

    db.session.commit()
    return redirect(url_for('collector.dashboard_collector'))
