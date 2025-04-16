from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from flask_login import login_required, current_user
from models import db, Pickup

user_bp = Blueprint('user', __name__)

@user_bp.route('/dashboard_user', methods=['GET', 'POST'])
@login_required
def dashboard_user():
    if not current_user.is_authenticated:
        return redirect(url_for('auth.login'))
        
    if request.method == 'POST':
        data = request.form
        new_pickup = Pickup(
            user_id=current_user.id,
            scrap_type=data.get('scrap_type'),
            weight=data.get('weight'),
            address=data.get('pickup_address'),
            date=data.get('date'),
            time=data.get('time'),
            status='Pending',
            estimated_price=data.get('estimated_price')
        )
        db.session.add(new_pickup)
        db.session.commit()
        flash('Pickup scheduled successfully!', 'success')
        return redirect(url_for('user.dashboard_user'))

    return render_template('dashboard_user.html', 
                         user_name=current_user.name, 
                         user_address=current_user.address)
