from flask import Blueprint, render_template, redirect, url_for
from flask_login import login_required, current_user
from models import db, Pickup, User

collector_bp = Blueprint('collector', __name__)

@collector_bp.route('/dashboard_collector')
@login_required
def dashboard_collector():
    if not current_user.is_authenticated or current_user.role != 'collector':
        return redirect(url_for('auth.login'))

    pickups = Pickup.query.filter(
        (Pickup.status == 'Pending') |
        ((Pickup.collector_id == current_user.id) & (Pickup.status.in_(['Accepted', 'Completed'])))
    ).all()

    return render_template('dashboard_collector.html', 
                         collector_name=current_user.name, 
                         pickups=pickups)
